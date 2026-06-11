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


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider 'nope'"):
        build_llm_client(_config("nope"))
