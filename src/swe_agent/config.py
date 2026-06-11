"""Configuration loading.

Reads ``config.toml`` (via the stdlib ``tomllib`` — no YAML dependency) and
pulls the API key from the environment by the name given in the file. The key
value never lives in the config file or in version control.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repo root is two levels up from this file: <root>/src/swe_agent/config.py
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


@dataclass(frozen=True)
class RateLimitConfig:
    """Knobs that keep us under a free-tier API's request limits."""

    delay_between_calls_s: float = 2.0
    max_retries: int = 5
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 60.0


@dataclass(frozen=True)
class LLMConfig:
    """Everything needed to construct and drive an :class:`LLMClient`."""

    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    def __repr__(self) -> str:  # never leak the secret in logs/tracebacks
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, api_key=***)"
        )


def load_config(
    path: Path | None = None,
    *,
    require_api_key: bool = True,
) -> LLMConfig:
    """Load :class:`LLMConfig` from a TOML file plus the environment.

    Args:
        path: config file path; defaults to ``<repo>/config.toml``.
        require_api_key: if True (default), raise when the API key env var is
            unset. Tests pass False to build a config without a real key.

    Raises:
        RuntimeError: if ``require_api_key`` and the key is missing.
    """
    load_dotenv()  # populate os.environ from a local .env if present (no-op in CI)
    path = path or DEFAULT_CONFIG_PATH

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    llm = raw["llm"]
    rl = llm.get("rate_limit", {})

    api_key_env = llm.get("api_key_env", "GROQ_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if require_api_key and not api_key:
        raise RuntimeError(
            f"Missing API key: environment variable {api_key_env!r} is not set. "
            f"Copy .env.example to .env and add your key, or export it."
        )

    return LLMConfig(
        provider=llm["provider"],
        model=llm["model"],
        api_key=api_key,
        base_url=llm.get("base_url"),
        temperature=llm.get("temperature", 0.0),
        max_tokens=llm.get("max_tokens", 4096),
        rate_limit=RateLimitConfig(
            delay_between_calls_s=rl.get("delay_between_calls_s", 2.0),
            max_retries=rl.get("max_retries", 5),
            initial_backoff_s=rl.get("initial_backoff_s", 2.0),
            max_backoff_s=rl.get("max_backoff_s", 60.0),
        ),
    )
