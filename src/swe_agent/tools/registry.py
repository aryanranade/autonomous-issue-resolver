"""Tool registry: bundle tools, advertise their specs, dispatch calls by name.

The agent loop (Phase 2) uses exactly two things from here: ``specs()`` to tell
the model what it can call, and ``dispatch()`` to run a model-chosen tool. All
exceptions are converted to error :class:`ToolResult`s so a bad tool call never
crashes the loop — the model gets the error text and can recover.
"""

from __future__ import annotations

from typing import Any

from swe_agent.llm.base import ToolSpec
from swe_agent.tools.base import Tool, ToolContext, ToolResult
from swe_agent.tools.filesystem import EditFile, ListDir, ReadFile
from swe_agent.tools.search import SearchCode
from swe_agent.tools.shell import RunShell, RunTests


class ToolRegistry:
    """Name-indexed collection of tools."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        """The tool declarations to advertise to the model."""
        return [tool.spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def dispatch(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        """Run tool ``name`` with ``args``; never raises."""
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools))
            return ToolResult.error(f"unknown tool {name!r}. Available: {known}")
        try:
            return tool.run(ctx, args)
        except ValueError as exc:
            # Expected validation problems (bad/missing args, path escape).
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 — last-resort guard for the loop
            return ToolResult.error(f"{type(exc).__name__}: {exc}")


def default_registry() -> ToolRegistry:
    """The standard five-tool set used by the agent."""
    return ToolRegistry(
        [
            ReadFile(),
            ListDir(),
            SearchCode(),
            EditFile(),
            RunShell(),
            RunTests(),
        ]
    )
