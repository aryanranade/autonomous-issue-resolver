"""Tests for the agent loop, driven by a scripted (offline) LLM client."""

from __future__ import annotations

from typing import Any

from swe_agent.agent.loop import Agent
from swe_agent.agent.result import StopReason
from swe_agent.config import AgentConfig
from swe_agent.llm.base import LLMClient, LLMResponse, Message, ToolSpec
from swe_agent.task import Task
from swe_agent.tools.base import ToolContext
from swe_agent.tools.registry import default_registry
from tests.conftest import ScriptedLLMClient, text_response, tool_response

FIX_ARGS = {
    "path": "calculator/ops.py",
    "old_string": "    return a + b  # BUG: should be a - b",
    "new_string": "    return a - b",
}


def _agent(client: ScriptedLLMClient, ctx: ToolContext, max_steps: int = 25) -> Agent:
    return Agent(
        llm=client,
        registry=default_registry(),
        ctx=ctx,
        config=AgentConfig(max_steps=max_steps),
    )


def test_happy_path_investigate_plan_fix_verify_finish(tool_ctx: ToolContext) -> None:
    client = ScriptedLLMClient(
        [
            tool_response("search_code", {"pattern": "def subtract"}),
            tool_response("read_file", {"path": "calculator/ops.py"}),
            tool_response(
                "record_plan",
                {
                    "root_cause": "subtract() adds instead of subtracting",
                    "files_to_change": ["calculator/ops.py"],
                    "approach": "change + to -",
                },
            ),
            tool_response("edit_file", FIX_ARGS),
            tool_response("run_tests", {}),
            tool_response("finish", {"summary": "Fixed subtract to use minus."}),
        ]
    )
    result = _agent(client, tool_ctx).run(Task(id="t1", problem_statement="subtract is wrong"))

    assert result.finished is True
    assert result.stop_reason is StopReason.FINISHED
    assert result.steps == 6
    assert result.summary == "Fixed subtract to use minus."
    # the bug was actually fixed on disk
    assert "return a - b" in (tool_ctx.root / "calculator" / "ops.py").read_text()
    # plan captured
    assert result.plan is not None
    assert result.plan.files_to_change == ["calculator/ops.py"]
    # tool records exclude the intercepted control tools (plan/finish)
    names = [r.name for r in result.tool_calls]
    assert names == ["search_code", "read_file", "edit_file", "run_tests"]


def test_runs_out_of_steps(tool_ctx: ToolContext) -> None:
    # A client that always asks to search -> never finishes.
    client = ScriptedLLMClient(
        [], default=tool_response("search_code", {"pattern": "x"})
    )
    result = _agent(client, tool_ctx, max_steps=3).run(
        Task(id="t2", problem_statement="loops forever")
    )
    assert result.finished is False
    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 3


def test_text_only_response_is_nudged(tool_ctx: ToolContext) -> None:
    client = ScriptedLLMClient(
        [
            text_response("Let me think about this..."),  # no tool call
            tool_response("finish", {"summary": "done"}),
        ]
    )
    result = _agent(client, tool_ctx).run(Task(id="t3", problem_statement="x"))

    assert result.finished is True
    assert result.steps == 2
    # the second call's message history must contain the nudge we injected
    second_call_msgs = client.calls[1]
    assert any(
        m.role == "user" and "Call a tool" in (m.content or "")
        for m in second_call_msgs
    )


def test_tool_error_is_recorded_and_loop_continues(tool_ctx: ToolContext) -> None:
    client = ScriptedLLMClient(
        [
            tool_response("read_file", {"path": "does_not_exist.py"}),  # error
            tool_response("finish", {"summary": "gave up cleanly"}),
        ]
    )
    result = _agent(client, tool_ctx).run(Task(id="t4", problem_statement="x"))

    assert result.finished is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].ok is False
    assert "file not found" in result.tool_calls[0].output_preview


def test_plan_accepts_comma_separated_files(tool_ctx: ToolContext) -> None:
    client = ScriptedLLMClient(
        [
            tool_response(
                "record_plan",
                {
                    "root_cause": "rc",
                    "files_to_change": "a.py, b.py",  # string, not array
                    "approach": "ap",
                },
            ),
            tool_response("finish", {"summary": "s"}),
        ]
    )
    result = _agent(client, tool_ctx).run(Task(id="t5", problem_statement="x"))
    assert result.plan is not None
    assert result.plan.files_to_change == ["a.py", "b.py"]


def test_patch_captures_source_change_excluding_pycache(git_repo) -> None:
    from pathlib import Path

    from swe_agent.tools.shell import LocalExecutor

    ctx = ToolContext(root=git_repo, executor=LocalExecutor())
    client = ScriptedLLMClient(
        [
            tool_response("edit_file", FIX_ARGS),
            tool_response("run_tests", {}),  # creates __pycache__/*.pyc
            tool_response("finish", {"summary": "fixed"}),
        ]
    )
    result = _agent(client, ctx).run(Task(id="g1", problem_statement="x"))

    assert result.made_changes
    assert "calculator/ops.py" in result.patch
    assert "return a - b" in result.patch
    # build artifacts from running tests must NOT appear in the patch
    assert "__pycache__" not in result.patch
    assert ".pyc" not in result.patch


class _FailingLLMClient(LLMClient):
    """Replays scripted responses, then raises — simulates the LLM dying mid-run."""

    def __init__(self, responses: list[LLMResponse], exc: Exception) -> None:
        self._responses = list(responses)
        self._exc = exc

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **overrides: Any,
    ) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        raise self._exc


def test_llm_failure_ends_run_with_error(tool_ctx: ToolContext) -> None:
    # A quota-exhausted / unavailable LLM must not crash the loop.
    client = _FailingLLMClient([], RuntimeError("boom: tokens-per-day exhausted"))
    result = _agent(client, tool_ctx, max_steps=5).run(
        Task(id="e1", problem_statement="x")
    )
    assert result.finished is False
    assert result.stop_reason is StopReason.ERROR
    assert result.error is not None and "boom" in result.error
    assert result.steps == 1  # failed on the very first LLM call


def test_llm_failure_after_edit_still_captures_partial_patch(git_repo) -> None:
    from swe_agent.tools.shell import LocalExecutor

    ctx = ToolContext(root=git_repo, executor=LocalExecutor())
    # Make a real edit, then the next LLM call dies.
    client = _FailingLLMClient(
        [tool_response("edit_file", FIX_ARGS)],
        RuntimeError("network down"),
    )
    result = _agent(client, ctx).run(Task(id="e2", problem_statement="x"))

    assert result.stop_reason is StopReason.ERROR
    assert result.error is not None
    # The edit made before the failure is still captured for grading.
    assert result.made_changes
    assert "return a - b" in result.patch


def test_transcript_threads_tool_results_to_each_call(tool_ctx: ToolContext) -> None:
    """Every assistant tool_call must be answered by a tool message (API rule)."""
    client = ScriptedLLMClient(
        [
            tool_response("list_dir", {}, call_id="call_A"),
            tool_response("finish", {"summary": "s"}, call_id="call_B"),
        ]
    )
    result = _agent(client, tool_ctx).run(Task(id="t6", problem_statement="x"))

    tool_msgs = [m for m in result.transcript if m.role == "tool"]
    answered_ids = {m.tool_call_id for m in tool_msgs}
    assert "call_A" in answered_ids  # list_dir result threaded back
