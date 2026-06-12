"""The agent loop: plan -> choose tool -> act -> observe -> iterate.

The loop is intentionally provider-agnostic and tool-agnostic: it depends only
on the LLMClient interface and a ToolRegistry, so the same loop runs against any
model (via config) and any tool set.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from swe_agent.agent.prompts import (
    FINISH_SPEC,
    RECORD_PLAN_SPEC,
    build_task_messages,
)
from swe_agent.agent.result import (
    AgentResult,
    Plan,
    StopReason,
    ToolCallRecord,
)
from swe_agent.config import AgentConfig
from swe_agent.llm.base import LLMClient, Message
from swe_agent.task import Task
from swe_agent.tools.base import ToolContext
from swe_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# A no-op reporter; the CLI passes ``print`` to stream progress.
Reporter = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _parse_plan(args: dict[str, Any]) -> Plan:
    """Build a Plan from record_plan arguments, tolerating sloppy shapes.

    Weak models sometimes send files_to_change as a comma-separated string
    instead of an array; accept both.
    """
    raw_files = args.get("files_to_change", [])
    if isinstance(raw_files, str):
        files = [f.strip() for f in raw_files.replace("\n", ",").split(",") if f.strip()]
    elif isinstance(raw_files, list):
        files = [str(f) for f in raw_files]
    else:
        files = []
    return Plan(
        root_cause=str(args.get("root_cause", "")).strip(),
        files_to_change=files,
        approach=str(args.get("approach", "")).strip(),
    )


def _brief(args: dict[str, Any]) -> str:
    """One-line preview of tool arguments for the progress trace."""
    parts = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        if len(text) > 40:
            text = text[:37] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


class Agent:
    """Drives an LLM through tools to resolve a task."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        ctx: ToolContext,
        config: AgentConfig,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._ctx = ctx
        self._config = config
        # Real tools plus the two control tools the loop intercepts.
        self._specs = [*registry.specs(), RECORD_PLAN_SPEC, FINISH_SPEC]

    def run(self, task: Task, report: Reporter = _noop) -> AgentResult:
        messages = build_task_messages(task)
        plan: Plan | None = None
        summary: str | None = None
        tool_records: list[ToolCallRecord] = []
        finished = False
        stop_reason = StopReason.MAX_STEPS  # default if we exhaust the budget
        error: str | None = None
        steps = 0

        while steps < self._config.max_steps:
            steps += 1
            try:
                resp = self._llm.complete(messages, tools=self._specs)
            except Exception as exc:  # noqa: BLE001 — terminal LLM failure ends the run
                # The LLM is unavailable (quota exhausted after retries, network,
                # auth). End cleanly with ERROR rather than crashing — a batch run
                # must survive one instance dying, and any edits made so far are
                # still captured as a patch below.
                error = f"{type(exc).__name__}: {exc}"
                stop_reason = StopReason.ERROR
                logger.warning("LLM call failed at step %d; ending run: %s", steps, error)
                report(f"[step {steps}] LLM error, ending run: {str(exc)[:200]}")
                break
            messages.append(
                Message(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls,
                )
            )
            if resp.content and resp.content.strip():
                report(f"[step {steps}] thought: {resp.content.strip()[:400]}")

            if not resp.tool_calls:
                # The model talked without acting. Nudge it rather than stall;
                # the step budget still bounds this.
                messages.append(
                    Message(
                        role="user",
                        content="Continue. Call a tool to make progress, or "
                        "call finish if the issue is resolved.",
                    )
                )
                continue

            for call in resp.tool_calls:
                if call.name == FINISH_SPEC.name:
                    summary = str(call.arguments.get("summary", "")).strip()
                    finished = True
                    stop_reason = StopReason.FINISHED
                    report(f"[step {steps}] finish: {summary[:200]}")
                    messages.append(
                        Message(role="tool", content="ok",
                                tool_call_id=call.id, name=call.name)
                    )
                    break

                if call.name == RECORD_PLAN_SPEC.name:
                    plan = _parse_plan(call.arguments)
                    report(f"[step {steps}] plan: {plan.root_cause[:160]}")
                    messages.append(
                        Message(role="tool", content="Plan recorded.",
                                tool_call_id=call.id, name=call.name)
                    )
                    continue

                # A real tool: dispatch via the registry (never raises).
                result = self._registry.dispatch(call.name, call.arguments, self._ctx)
                tool_records.append(
                    ToolCallRecord(
                        step=steps,
                        name=call.name,
                        arguments=call.arguments,
                        ok=result.ok,
                        output_preview=result.output[:300],
                    )
                )
                status = "ok" if result.ok else "ERROR"
                report(f"[step {steps}] {call.name}({_brief(call.arguments)}) -> {status}")
                messages.append(
                    Message(role="tool", content=result.output,
                            tool_call_id=call.id, name=call.name)
                )

            if finished:
                break

        patch = self._compute_patch()
        return AgentResult(
            task_id=task.id,
            stop_reason=stop_reason,
            finished=finished,
            steps=steps,
            plan=plan,
            summary=summary,
            patch=patch,
            tool_calls=tool_records,
            transcript=messages,
            error=error,
        )

    def _compute_patch(self) -> str:
        """Capture the working-tree diff as a unified patch.

        ``git add -A -N`` registers new files as intent-to-add so they appear in
        ``git diff`` too. We exclude build artifacts (``__pycache__``/``*.pyc``)
        that running the tests creates — otherwise they pollute the patch and it
        won't apply cleanly to a fresh checkout. Returns "" when the root isn't a
        git repo (the diff produces no stdout), so this is safe to call always.
        """
        result = self._ctx.executor.run(
            "git add -A -N >/dev/null 2>&1 && "
            "git diff -- ':(exclude)**/__pycache__/**' ':(exclude)**/*.pyc'",
            cwd=self._ctx.root,
            timeout=30,
        )
        return result.stdout
