"""Foundations for the tool layer.

Defines the contract every tool implements, the result type fed back to the
agent, the per-run context (repo root + command executor), and path-containment
helpers so filesystem tools can't escape the repository.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_agent.llm.base import ToolSpec


@dataclass
class ToolResult:
    """Outcome of running a tool.

    ``output`` is the text that gets handed back to the model as the tool
    message, so it must be human/LLM-readable. ``ok`` lets the agent loop
    distinguish a clean result from a recoverable failure without parsing text.
    """

    ok: bool
    output: str

    @classmethod
    def success(cls, output: str) -> "ToolResult":
        return cls(ok=True, output=output)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        # Prefix so the model clearly sees this turn failed and can adjust.
        return cls(ok=False, output=f"ERROR: {message}")


@dataclass
class ExecResult:
    """Result of running a shell command."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandExecutor(ABC):
    """Runs shell commands. Abstract so the backend can change per phase.

    Phase 1 ships ``LocalExecutor`` (subprocess on the host). Phase 3 will add a
    ``DockerExecutor`` that runs the same commands inside a per-task container;
    tools won't change because they only depend on this interface.
    """

    @abstractmethod
    def run(self, command: str, *, cwd: Path, timeout: int) -> ExecResult:
        raise NotImplementedError


@dataclass
class ToolContext:
    """Everything a tool needs at run time, independent of the LLM."""

    root: Path
    executor: CommandExecutor
    max_output_chars: int = 10_000  # token-budget guard on any single tool output


class Tool(ABC):
    """A capability the agent can invoke.

    Each tool exposes a :class:`ToolSpec` (advertised to the model) and a
    ``run`` that takes the raw argument dict the model produced.

    Contract: ``run`` returns a :class:`ToolResult` for *runtime* conditions the
    model should see and react to (file not found, command failed, no matches).
    It may *raise* ValueError for invalid arguments (missing/wrong-typed args,
    a path that escapes the root). The agent never calls ``run`` directly — it
    goes through :meth:`ToolRegistry.dispatch`, which converts any raised
    exception into an error :class:`ToolResult`, so the loop never crashes.
    """

    #: stable identifier the model uses to call the tool
    name: str

    @abstractmethod
    def spec(self) -> ToolSpec:
        """JSON-schema declaration advertised to the model."""
        raise NotImplementedError

    @abstractmethod
    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        """Execute with the model-supplied ``args`` dict."""
        raise NotImplementedError


def resolve_in_root(root: Path, relpath: str) -> Path:
    """Resolve ``relpath`` against ``root`` and refuse to escape it.

    Returns the absolute resolved path. Raises ValueError if the result would
    fall outside ``root`` (e.g. ``../../etc/passwd``).
    """
    root = root.resolve()
    candidate = (root / relpath).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path {relpath!r} escapes the repository root")
    return candidate


def truncate(text: str, max_chars: int) -> str:
    """Middle-truncate ``text`` so both head and tail survive.

    Middle (not tail) truncation matters for test output: pytest's failing-test
    list is near the top and its summary is at the very bottom — we want both.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [truncated {omitted} chars] ...\n{text[-half:]}"


def require_str(args: dict[str, Any], key: str) -> str:
    """Pull a required string argument or raise a clear ValueError."""
    if key not in args:
        raise ValueError(f"missing required argument {key!r}")
    value = args[key]
    if not isinstance(value, str):
        raise ValueError(f"argument {key!r} must be a string, got {type(value).__name__}")
    return value
