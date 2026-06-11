"""Structured output of an agent run.

Everything here is designed to be logged and analyzed later (Phase 4 records it,
Phase 5 categorizes failures from it), so it captures not just success/failure
but *how* the run unfolded: the plan, every tool call, the final diff, and the
full transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from swe_agent.llm.base import Message


class StopReason(str, Enum):
    """Why the loop ended. str-valued so it serializes cleanly to JSON."""

    FINISHED = "finished"      # model called finish
    MAX_STEPS = "max_steps"    # ran out of step budget
    ERROR = "error"            # unrecoverable error in the loop itself


@dataclass
class Plan:
    """The diagnosis + intent the model recorded before editing."""

    root_cause: str
    files_to_change: list[str]
    approach: str


@dataclass
class ToolCallRecord:
    """One tool invocation, captured for analysis."""

    step: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    output_preview: str


@dataclass
class AgentResult:
    task_id: str
    stop_reason: StopReason
    finished: bool
    steps: int
    plan: Plan | None
    summary: str | None
    patch: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)

    @property
    def made_changes(self) -> bool:
        """True if the run produced a non-empty diff."""
        return bool(self.patch.strip())
