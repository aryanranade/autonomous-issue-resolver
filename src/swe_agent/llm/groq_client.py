"""Groq implementation of :class:`LLMClient`.

Groq exposes an OpenAI-compatible endpoint, so we reuse the ``openai`` SDK
pointed at Groq's ``base_url``. This module is the *only* place that knows about
OpenAI/Groq wire formats — it translates our neutral dataclasses to/from the
SDK's shapes. A future provider with a different API implements the same
``LLMClient`` contract here and the agent loop never notices.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from swe_agent.config import LLMConfig
from swe_agent.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from swe_agent.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# Transient failures worth retrying. Note: 4xx like auth/bad-request are NOT
# here on purpose — retrying them just wastes the rate-limit budget.
_RETRYABLE: tuple[type[BaseException], ...] = (
    RateLimitError,       # HTTP 429 — the free-tier limit we most expect
    APITimeoutError,
    APIConnectionError,
    InternalServerError,  # HTTP 5xx
)


def _message_to_dict(m: Message) -> dict[str, Any]:
    """Neutral Message -> OpenAI chat message dict."""
    d: dict[str, Any] = {"role": m.role}
    # content can legitimately be None on an assistant turn that only has
    # tool_calls; include the key so the API sees an explicit null.
    d["content"] = m.content
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        d["name"] = m.name
    return d


def _tool_to_dict(t: ToolSpec) -> dict[str, Any]:
    """Neutral ToolSpec -> OpenAI tool dict."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


class GroqClient(LLMClient):
    """LLMClient backed by Groq's OpenAI-compatible API."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Args:
            config: provider/model/rate-limit settings.
            client: pre-built OpenAI SDK client; injectable for tests.
            sleep: sleep function; injectable so tests don't actually wait.
        """
        self._config = config
        self._sleep = sleep
        # The SDK's own retry layer is the first line of defence against 429s and
        # honours Groq's Retry-After header (better than blind backoff). We set it
        # from config so it isn't the SDK default of 2; retry_with_backoff below
        # is the outer safety net if the SDK still gives up.
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=config.rate_limit.max_retries,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **overrides: Any,
    ) -> LLMResponse:
        cfg = self._config
        payload: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
            "messages": [_message_to_dict(m) for m in messages],
            "temperature": overrides.get("temperature", cfg.temperature),
            "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
        }
        if tools:
            payload["tools"] = [_tool_to_dict(t) for t in tools]
            payload["tool_choice"] = overrides.get("tool_choice", "auto")

        rl = cfg.rate_limit
        completion = retry_with_backoff(
            lambda: self._client.chat.completions.create(**payload),
            retry_on=_RETRYABLE,
            max_retries=rl.max_retries,
            initial_backoff_s=rl.initial_backoff_s,
            max_backoff_s=rl.max_backoff_s,
            sleep=self._sleep,
        )

        # Fixed inter-call throttle: the simplest reliable way to stay under a
        # requests-per-minute cap during a long benchmark run.
        if rl.delay_between_calls_s > 0:
            self._sleep(rl.delay_between_calls_s)

        return self._parse(completion)

    @staticmethod
    def _parse(completion: Any) -> LLMResponse:
        """OpenAI completion object -> neutral LLMResponse."""
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # Models occasionally emit malformed JSON; surface it rather
                # than crash, so the agent loop can decide how to recover.
                logger.warning("Tool call %s had non-JSON arguments", tc.function.name)
                args = {"__raw__": raw_args}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        usage = Usage()
        if getattr(completion, "usage", None) is not None:
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
                total_tokens=completion.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw=completion,
        )
