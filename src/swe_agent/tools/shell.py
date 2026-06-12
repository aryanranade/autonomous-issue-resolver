"""Command execution: LocalExecutor + run_shell / run_tests tools.

LocalExecutor runs commands on the host (Phase 1). In Phase 3 a DockerExecutor
implements the same ``CommandExecutor`` interface so the *same* tools run inside
an isolated per-task container. Until then there is NO sandboxing — run_shell
executes arbitrary commands on this machine; that's acceptable only because
Phase 1 testing drives it with controlled commands against a temp directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from swe_agent.llm.base import ToolSpec
from swe_agent.tools.base import (
    CommandExecutor,
    ExecResult,
    Tool,
    ToolContext,
    ToolResult,
    require_str,
    truncate,
)

# Upper bound so a runaway command can't hang a benchmark run forever.
_MAX_TIMEOUT_S = 600


class LocalExecutor(CommandExecutor):
    """Run a command via the shell on the host, capturing output."""

    def run(self, command: str, *, cwd: Path, timeout: int) -> ExecResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,  # conventional timeout code
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "" if isinstance(exc.stderr, str) else "")
                + f"\n[timed out after {timeout}s]",
                timed_out=True,
            )
        return ExecResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


def _format(result: ExecResult, max_chars: int) -> str:
    """Render an ExecResult into the text the model sees."""
    half = max(1, max_chars // 2)
    parts = [f"exit_code: {result.exit_code}"]
    if result.timed_out:
        parts.append("(timed out)")
    parts.append(f"--- stdout ---\n{truncate(result.stdout, half)}")
    parts.append(f"--- stderr ---\n{truncate(result.stderr, half)}")
    return "\n".join(parts)


class RunShell(Tool):
    """Run an arbitrary shell command in the repository root."""

    name = "run_shell"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Run a shell command from the repository root and return its exit "
                "code, stdout, and stderr. Use for building, installing, or "
                "inspecting; prefer run_tests for running the test suite."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Max seconds before the command is killed "
                        f"(capped at {_MAX_TIMEOUT_S}). Default 60.",
                    },
                },
                "required": ["command"],
            },
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        command = require_str(args, "command")
        timeout = min(int(args.get("timeout", 60)), _MAX_TIMEOUT_S)
        result = ctx.executor.run(command, cwd=ctx.root, timeout=timeout)
        output = _format(result, ctx.max_output_chars)
        # Non-zero exit is reported as output, not a tool error: the model often
        # *wants* to see failing test/build output and act on it.
        return ToolResult.success(output)


class RunTests(Tool):
    """Run the test suite (pytest by default) and return the result."""

    name = "run_tests"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Run the project's tests with pytest and return exit code and "
                "output. Optionally target a specific test file, directory, or "
                "node id to run a focused subset."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Optional pytest target (path or node id). "
                        "Omit to run the whole suite.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Max seconds before tests are killed "
                        f"(capped at {_MAX_TIMEOUT_S}). Default 300.",
                    },
                },
                "required": [],
            },
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        target = args.get("target", "")
        if not isinstance(target, str):
            return ToolResult.error("argument 'target' must be a string")
        timeout = min(int(args.get("timeout", 300)), _MAX_TIMEOUT_S)
        # Use the configured interpreter so the right venv/pytest is selected.
        # On the host this is sys.executable; in a container it's the activated
        # conda env's "python" (see ToolContext.python_executable).
        command = f"{ctx.python_executable} -m pytest -q"
        if target.strip():
            command += f" {target.strip()}"
        result = ctx.executor.run(command, cwd=ctx.root, timeout=timeout)
        return ToolResult.success(_format(result, ctx.max_output_chars))
