"""Tests for the LLM client factory."""

from __future__ import annotations

import pytest

from swe_agent.config import LLMConfig
from swe_agent.llm.factory import build_llm_client
from swe_agent.llm.groq_client import GroqClient


def _config(provider: str) -> LLMConfig:
    return LLMConfig(provider=provider, model="m", api_key="x", base_url="http://t/v1")


def test_builds_groq_client() -> None:
    client = build_llm_client(_config("groq"))
    assert isinstance(client, GroqClient)


@pytest.mark.parametrize("provider", ["groq", "gemini", "openai"])
def test_openai_compatible_providers_all_build(provider: str) -> None:
    """Every OpenAI-compatible provider name resolves to the generic client."""
    assert isinstance(build_llm_client(_config(provider)), GroqClient)


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider 'nope'"):
        build_llm_client(_config("nope"))


def test_unknown_provider_message_lists_known_providers() -> None:
    """The error has to be actionable — it's the first thing a new user hits."""
    with pytest.raises(ValueError, match="config.toml") as excinfo:
        build_llm_client(_config("nope"))
    assert "openai" in str(excinfo.value)


def test_anthropic_is_rejected_with_a_specific_explanation() -> None:
    """Claude is not OpenAI-compatible; failing at startup beats failing mid-run."""
    with pytest.raises(ValueError, match="not OpenAI-compatible"):
        build_llm_client(_config("anthropic"))
