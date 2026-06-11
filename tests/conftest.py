"""Shared pytest fixtures and lightweight fakes.

We never hit the real Groq API in tests. Instead we build fake OpenAI-SDK-shaped
response objects with ``types.SimpleNamespace`` and inject a fake client.

This module also provides a small on-disk "dummy repo" and a ToolContext bound
to it, used by the Phase 1 tool tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swe_agent.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage
from swe_agent.tools.base import ToolContext
from swe_agent.tools.shell import LocalExecutor


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


# --------------------------------------------------------------------------- #
# Dummy repo + ToolContext fixtures (Phase 1 tool tests)
# --------------------------------------------------------------------------- #

# A deliberately buggy mini-package: subtract() adds instead of subtracts, so
# tests/test_ops.py has one passing and one failing test. Lets us exercise
# read/search/edit and then run_tests going red -> green after a fix.
_OPS_PY = '''\
"""Tiny calculator used by the tool tests."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a + b  # BUG: should be a - b
'''

_TEST_OPS_PY = '''\
from calculator.ops import add, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2
'''


@pytest.fixture
def dummy_repo(tmp_path: Path) -> Path:
    """Build a small repo on disk and return its root path."""
    pkg = tmp_path / "calculator"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ops.py").write_text(_OPS_PY)

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ops.py").write_text(_TEST_OPS_PY)

    (tmp_path / "README.md").write_text("# Dummy\nA tiny calculator package.\n")
    # A directory that searches should skip.
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("subtract should not be found here")
    # A binary file searches should skip.
    (tmp_path / "data.bin").write_bytes(b"\x00\x01subtract\x00\x02")
    return tmp_path


@pytest.fixture
def tool_ctx(dummy_repo: Path) -> ToolContext:
    """ToolContext bound to the dummy repo with a real local executor."""
    return ToolContext(root=dummy_repo, executor=LocalExecutor())


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real git repo with the buggy calculator, committed at a clean baseline.

    Separate from ``dummy_repo`` (which has a *fake* .git for the search test) so
    patch-capture can be tested against actual ``git diff`` output.
    """
    pkg = tmp_path / "calculator"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ops.py").write_text(_OPS_PY)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ops.py").write_text(_TEST_OPS_PY)

    for cmd in (
        "git init -q",
        "git add -A",
        "git -c user.name=t -c user.email=t@t commit -qm baseline",
    ):
        subprocess.run(cmd, shell=True, cwd=tmp_path, check=True)
    return tmp_path


# --------------------------------------------------------------------------- #
# Scripted LLM client (Phase 2 agent-loop tests)
# --------------------------------------------------------------------------- #


def tool_response(name: str, arguments: dict[str, Any], call_id: str = "c") -> LLMResponse:
    """An LLMResponse that requests a single tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        finish_reason="tool_calls",
        usage=Usage(),
    )


def text_response(text: str) -> LLMResponse:
    """An LLMResponse with plain text and no tool calls."""
    return LLMResponse(content=text, tool_calls=[], finish_reason="stop", usage=Usage())


class ScriptedLLMClient(LLMClient):
    """LLMClient that replays a fixed list of responses, for deterministic tests.

    Once the scripted list is exhausted it returns ``default`` repeatedly (used
    to simulate a model that never stops). Records the messages seen on each call.
    """

    def __init__(
        self, responses: list[LLMResponse], default: LLMResponse | None = None
    ) -> None:
        self._responses = list(responses)
        self._default = default
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **overrides: Any,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        if self._default is not None:
            return self._default
        raise AssertionError("ScriptedLLMClient ran out of responses")
