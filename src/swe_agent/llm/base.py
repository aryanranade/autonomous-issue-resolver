"""Provider-neutral LLM interface.

This module defines the *contract* every LLM provider must satisfy, expressed
in our own dataclasses — deliberately NOT in any vendor's request/response
types. The whole point: the rest of the codebase (agent loop, tools) depends
only on these abstractions, so swapping Groq for another provider later means
writing one new `LLMClient` subclass, not touching call sites.

Translation between these neutral types and a specific vendor's wire format
lives with that vendor's client (e.g. ``groq_client.py``), never here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Declaration of a tool the model is allowed to call.

    ``parameters`` is a JSON Schema object describing the tool's arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One turn in a conversation.

    ``role`` is one of: ``system``, ``user``, ``assistant``, ``tool``.
    - assistant turns may carry ``tool_calls``.
    - ``tool`` turns carry the result of a call and must set ``tool_call_id``.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # required when role == "tool"
    name: str | None = None          # optional tool/function name on tool turns


@dataclass
class Usage:
    """Token accounting for a single completion (useful for cost/limit tracking)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Normalized result of one completion call."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: Usage
    raw: Any = None  # the untouched provider response, for debugging only


class LLMClient(ABC):
    """The single interface the rest of the agent depends on."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **overrides: Any,
    ) -> LLMResponse:
        """Run one completion.

        Args:
            messages: conversation so far, oldest first.
            tools: tools the model may call this turn (None = plain chat).
            **overrides: per-call overrides of config defaults
                (e.g. ``temperature``, ``max_tokens``, ``tool_choice``).

        Returns:
            A normalized :class:`LLMResponse`.
        """
        raise NotImplementedError
