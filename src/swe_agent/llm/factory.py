"""Factory that builds an :class:`LLMClient` from config.

This is the single switch point keyed on ``config.provider``. Adding a provider
= add one entry to ``_REGISTRY``; callers keep calling ``build_llm_client``.
"""

from __future__ import annotations

from collections.abc import Callable

from swe_agent.config import LLMConfig
from swe_agent.llm.base import LLMClient
from swe_agent.llm.groq_client import GroqClient

# provider name (from config.toml) -> factory callable.
# Typed as a callable (not `type[LLMClient]`) so each implementation is free to
# take extra keyword-only args (e.g. injectable client/sleep) beyond the config.
_REGISTRY: dict[str, Callable[[LLMConfig], LLMClient]] = {
    "groq": GroqClient,
    # GroqClient is a generic OpenAI-compatible client (it only depends on
    # base_url + api key), so it also drives other OpenAI-compatible endpoints
    # like Gemini's — the difference is entirely in config.toml. `openai` is the
    # catch-all name for any such endpoint (OpenAI itself, OpenRouter, DeepSeek,
    # Together, a local vLLM/Ollama server, ...): set `base_url` and go.
    "gemini": GroqClient,
    "openai": GroqClient,
    # NOT registered on purpose: "anthropic". Claude is not an OpenAI-compatible
    # API — it needs a native client implementing LLMClient (see README
    # "Using a different provider"). Wiring it here would fail at request time
    # rather than at startup, which is the worse failure.
}


def build_llm_client(config: LLMConfig) -> LLMClient:
    """Construct the LLMClient named by ``config.provider``.

    Raises:
        ValueError: if the provider name is not registered.
    """
    try:
        cls = _REGISTRY[config.provider]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        hint = (
            "Set [llm] provider in config.toml to one of: "
            f"{known}. Any OpenAI-compatible endpoint works with "
            'provider = "openai" plus the right base_url.'
        )
        if config.provider == "anthropic":
            hint = (
                "Anthropic's API is not OpenAI-compatible, so it needs a native "
                "client implementing LLMClient (see README, 'Using a different "
                "provider'). It is deliberately not aliased to the "
                "OpenAI-compatible client."
            )
        raise ValueError(
            f"Unknown LLM provider {config.provider!r}. {hint}"
        ) from None
    return cls(config)
