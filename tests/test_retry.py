"""Tests for the retry-with-backoff helper."""

from __future__ import annotations

import pytest

from swe_agent.utils.retry import retry_with_backoff


class Transient(Exception):
    pass


class Fatal(Exception):
    pass


def test_returns_on_first_success() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    assert retry_with_backoff(fn, retry_on=(Transient,), sleep=lambda _: None) == "ok"
    assert calls["n"] == 1


def test_retries_then_succeeds() -> None:
    seq = [Transient("boom"), Transient("boom"), "ok"]

    def fn() -> str:
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    slept: list[float] = []
    result = retry_with_backoff(
        fn,
        retry_on=(Transient,),
        initial_backoff_s=1.0,
        jitter=False,
        sleep=slept.append,
    )
    assert result == "ok"
    # Two failures -> two sleeps, exponential: 1.0 then 2.0
    assert slept == [1.0, 2.0]


def test_raises_after_exhausting_retries() -> None:
    def fn() -> str:
        raise Transient("always")

    with pytest.raises(Transient):
        retry_with_backoff(
            fn, retry_on=(Transient,), max_retries=3, sleep=lambda _: None
        )


def test_non_retryable_propagates_immediately() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise Fatal("nope")

    with pytest.raises(Fatal):
        retry_with_backoff(fn, retry_on=(Transient,), sleep=lambda _: None)
    assert calls["n"] == 1  # never retried


def test_backoff_is_capped() -> None:
    seq = [Transient()] * 4 + ["ok"]

    def fn() -> str:
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    slept: list[float] = []
    retry_with_backoff(
        fn,
        retry_on=(Transient,),
        initial_backoff_s=10.0,
        max_backoff_s=15.0,
        jitter=False,
        sleep=slept.append,
    )
    # 10 -> capped 15 -> 15 -> 15
    assert slept == [10.0, 15.0, 15.0, 15.0]
