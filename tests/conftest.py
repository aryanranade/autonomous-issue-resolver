"""Shared pytest fixtures and lightweight fakes.

We never hit the real Groq API in tests. Instead we build fake OpenAI-SDK-shaped
response objects with ``types.SimpleNamespace`` and inject a fake client.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def make_completion(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> SimpleNamespace:
    """Build an object shaped like an ``openai`` ChatCompletion.

    Only the attributes our ``GroqClient._parse`` reads are populated.
    """
    tc_objs = []
    for tc in tool_calls or []:
        tc_objs.append(
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc["arguments"],  # a JSON string, as the API sends
                ),
            )
        )
    usage = usage or {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tc_objs or None),
            )
        ],
        usage=SimpleNamespace(**usage),
    )


class FakeOpenAI:
    """Stand-in for ``openai.OpenAI``.

    Either returns ``responses`` in order, or raises the next item if it's an
    exception instance — lets us script retry scenarios. Records every payload
    passed to ``chat.completions.create`` in ``self.calls``.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def make_completion_fixture():
    return make_completion
