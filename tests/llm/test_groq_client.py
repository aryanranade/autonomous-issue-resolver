"""Tests for GroqClient: request translation, response parsing, retry, throttle.

All tests inject a FakeOpenAI client and a no-op/recording sleep, so nothing
touches the network or actually waits.
"""

from __future__ import annotations

import json

from openai import RateLimitError

from swe_agent.config import LLMConfig, RateLimitConfig
from swe_agent.llm.base import Message, ToolSpec
from swe_agent.llm.groq_client import GroqClient
from tests.conftest import FakeOpenAI, make_completion


def _config(**rl_overrides: object) -> LLMConfig:
    return LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key="x",
        base_url="https://example.test/v1",
        temperature=0.0,
        max_tokens=256,
        rate_limit=RateLimitConfig(
            delay_between_calls_s=1.0,
            max_retries=3,
            initial_backoff_s=1.0,
            max_backoff_s=5.0,
            **rl_overrides,  # type: ignore[arg-type]
        ),
    )


def _rate_limit_error() -> RateLimitError:
    # RateLimitError needs a response/body; construct a minimal valid one.
    import httpx

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def test_translates_messages_and_tools_into_payload() -> None:
    fake = FakeOpenAI([make_completion(content="hi")])
    slept: list[float] = []
    client = GroqClient(_config(), client=fake, sleep=slept.append)  # type: ignore[arg-type]

    tools = [
        ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    client.complete([Message(role="user", content="hello")], tools=tools)

    payload = fake.calls[0]
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 256
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert payload["tool_choice"] == "auto"


def test_no_tools_means_no_tool_keys() -> None:
    fake = FakeOpenAI([make_completion()])
    client = GroqClient(_config(), client=fake, sleep=lambda _: None)  # type: ignore[arg-type]
    client.complete([Message(role="user", content="hi")])
    assert "tools" not in fake.calls[0]
    assert "tool_choice" not in fake.calls[0]


def test_parses_plain_text_response() -> None:
    fake = FakeOpenAI([make_completion(content="the answer", finish_reason="stop")])
    client = GroqClient(_config(), client=fake, sleep=lambda _: None)  # type: ignore[arg-type]
    resp = client.complete([Message(role="user", content="q")])
    assert resp.content == "the answer"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens == 3


def test_parses_tool_calls_and_decodes_arguments() -> None:
    completion = make_completion(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "name": "edit_file",
                "arguments": json.dumps({"path": "a.py", "content": "x = 1"}),
            }
        ],
    )
    fake = FakeOpenAI([completion])
    client = GroqClient(_config(), client=fake, sleep=lambda _: None)  # type: ignore[arg-type]
    resp = client.complete([Message(role="user", content="fix it")])

    assert resp.content is None
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "edit_file"
    assert call.arguments == {"path": "a.py", "content": "x = 1"}


def test_malformed_tool_arguments_are_surfaced_not_crashed() -> None:
    completion = make_completion(
        content=None,
        tool_calls=[{"id": "c", "name": "edit_file", "arguments": "{not json"}],
    )
    fake = FakeOpenAI([completion])
    client = GroqClient(_config(), client=fake, sleep=lambda _: None)  # type: ignore[arg-type]
    resp = client.complete([Message(role="user", content="x")])
    assert resp.tool_calls[0].arguments == {"__raw__": "{not json"}


def test_retries_on_rate_limit_then_succeeds() -> None:
    fake = FakeOpenAI([_rate_limit_error(), make_completion(content="recovered")])
    slept: list[float] = []
    client = GroqClient(_config(), client=fake, sleep=slept.append)  # type: ignore[arg-type]
    resp = client.complete([Message(role="user", content="x")])

    assert resp.content == "recovered"
    assert len(fake.calls) == 2  # one retry happened
    # First sleep is the backoff (~1.0 + jitter); last is the inter-call delay (1.0).
    assert slept[-1] == 1.0
    assert slept[0] >= 1.0


def test_inter_call_delay_applied_on_success() -> None:
    fake = FakeOpenAI([make_completion()])
    slept: list[float] = []
    client = GroqClient(_config(), client=fake, sleep=slept.append)  # type: ignore[arg-type]
    client.complete([Message(role="user", content="x")])
    assert slept == [1.0]  # exactly the throttle delay, no retries


def test_assistant_tool_call_message_roundtrips_into_payload() -> None:
    """An assistant turn carrying tool_calls + a tool result turn serialize correctly."""
    from swe_agent.llm.base import ToolCall

    fake = FakeOpenAI([make_completion()])
    client = GroqClient(_config(), client=fake, sleep=lambda _: None)  # type: ignore[arg-type]
    messages = [
        Message(role="user", content="fix"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a"})],
        ),
        Message(role="tool", content="file contents", tool_call_id="c1", name="read_file"),
    ]
    client.complete(messages)

    sent = fake.calls[0]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"path": "a"})
    assert sent[2]["tool_call_id"] == "c1"
