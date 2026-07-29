"""Tests for LocalExecutor and the run_shell / run_tests tools."""

from __future__ import annotations

from swe_agent.tools.base import ToolContext
from swe_agent.tools.filesystem import EditFile
from dataclasses import replace

from swe_agent.tools.base import ExecResult
from swe_agent.tools.shell import LocalExecutor, RunShell, RunTests


def test_run_shell_captures_stdout_and_exit_code(tool_ctx: ToolContext) -> None:
    res = RunShell().run(tool_ctx, {"command": "echo hello"})
    assert res.ok
    assert "exit_code: 0" in res.output
    assert "hello" in res.output


def test_run_shell_nonzero_exit_is_reported_not_errored(tool_ctx: ToolContext) -> None:
    res = RunShell().run(tool_ctx, {"command": "exit 3"})
    # Non-zero exit is a successful tool call that reports a failing command.
    assert res.ok
    assert "exit_code: 3" in res.output


def test_run_shell_runs_in_repo_root(tool_ctx: ToolContext) -> None:
    res = RunShell().run(tool_ctx, {"command": "ls"})
    assert res.ok
    assert "calculator" in res.output
    assert "README.md" in res.output


def test_run_shell_timeout(tool_ctx: ToolContext) -> None:
    res = RunShell().run(tool_ctx, {"command": "sleep 5", "timeout": 1})
    assert res.ok
    assert "timed out" in res.output
    assert "exit_code: 124" in res.output


def test_local_executor_timeout_flag(dummy_repo) -> None:
    result = LocalExecutor().run("sleep 5", cwd=dummy_repo, timeout=1)
    assert result.timed_out
    assert result.exit_code == 124


def test_run_tests_red_then_green(tool_ctx: ToolContext) -> None:
    # The dummy repo ships with a failing test_subtract.
    red = RunTests().run(tool_ctx, {})
    assert red.ok
    assert "exit_code: 0" not in red.output  # at least one failure
    assert "test_subtract" in red.output

    # Fix the bug, then tests should pass.
    EditFile().run(
        tool_ctx,
        {
            "path": "calculator/ops.py",
            "old_string": "    return a + b  # BUG: should be a - b",
            "new_string": "    return a - b",
        },
    )
    green = RunTests().run(tool_ctx, {})
    assert green.ok
    assert "exit_code: 0" in green.output


def test_run_tests_targeted(tool_ctx: ToolContext) -> None:
    res = RunTests().run(tool_ctx, {"target": "tests/test_ops.py::test_add"})
    assert res.ok
    assert "exit_code: 0" in res.output


def test_run_tests_uses_the_repos_own_runner_when_supplied(
    tool_ctx: ToolContext,
) -> None:
    """69% of SWE-bench Lite isn't pytest, so an explicit runner must win.

    Guards the regression where run_tests hardcoded `python -m pytest`, which
    silently broke the agent's verify step on django/sympy/sphinx.
    """
    recorded: list[str] = []

    class Recorder:
        def run(self, command: str, *, cwd: object, timeout: int) -> object:
            recorded.append(command)
            return ExecResult(exit_code=0, stdout="ok", stderr="", timed_out=False)

    ctx = replace(tool_ctx, executor=Recorder(), test_command="bin/test -C --verbose")
    RunTests().run(ctx, {})

    assert recorded == ["bin/test -C --verbose"]
    assert "pytest" not in recorded[0]


def test_run_tests_appends_target_to_the_repo_runner(tool_ctx: ToolContext) -> None:
    recorded: list[str] = []

    class Recorder:
        def run(self, command: str, *, cwd: object, timeout: int) -> object:
            recorded.append(command)
            return ExecResult(exit_code=0, stdout="ok", stderr="", timed_out=False)

    ctx = replace(
        tool_ctx,
        executor=Recorder(),
        test_command="./tests/runtests.py --settings=test_sqlite",
    )
    RunTests().run(ctx, {"target": "test_utils.tests"})

    assert recorded == ["./tests/runtests.py --settings=test_sqlite test_utils.tests"]


def test_run_tests_falls_back_to_pytest_for_a_plain_local_repo(
    tool_ctx: ToolContext,
) -> None:
    """agent.cli points at arbitrary repos with no spec — pytest stays the default."""
    assert tool_ctx.test_command is None
    res = RunTests().run(tool_ctx, {})
    assert res.ok
