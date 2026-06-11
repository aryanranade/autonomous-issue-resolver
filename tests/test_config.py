"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from swe_agent.config import AgentConfig, LLMConfig, load_agent_config, load_config

SAMPLE = """\
[llm]
provider = "groq"
model = "llama-3.3-70b-versatile"
api_key_env = "MY_TEST_KEY"
base_url = "https://example.test/v1"
temperature = 0.2
max_tokens = 1234

[llm.rate_limit]
delay_between_calls_s = 0.5
max_retries = 7
initial_backoff_s = 1.5
max_backoff_s = 30.0
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(SAMPLE)
    return p


def test_loads_all_fields(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "secret-123")
    cfg = load_config(config_file)

    assert isinstance(cfg, LLMConfig)
    assert cfg.provider == "groq"
    assert cfg.model == "llama-3.3-70b-versatile"
    assert cfg.api_key == "secret-123"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 1234
    assert cfg.rate_limit.delay_between_calls_s == 0.5
    assert cfg.rate_limit.max_retries == 7
    assert cfg.rate_limit.initial_backoff_s == 1.5
    assert cfg.rate_limit.max_backoff_s == 30.0


def test_missing_key_raises(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MY_TEST_KEY"):
        load_config(config_file)


def test_require_api_key_false_allows_missing(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    cfg = load_config(config_file, require_api_key=False)
    assert cfg.api_key == ""


def test_repr_hides_secret(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "secret-123")
    cfg = load_config(config_file)
    assert "secret-123" not in repr(cfg)
    assert "api_key=***" in repr(cfg)


def test_load_agent_config_reads_max_steps(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[agent]\nmax_steps = 7\n")
    cfg = load_agent_config(p)
    assert isinstance(cfg, AgentConfig)
    assert cfg.max_steps == 7


def test_load_agent_config_defaults_when_section_missing(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[llm]\nprovider='groq'\nmodel='m'\n")
    cfg = load_agent_config(p)
    assert cfg.max_steps == 25  # default
