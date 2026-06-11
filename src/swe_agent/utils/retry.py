"""Generic retry-with-exponential-backoff helper.

Hand-rolled (no ``tenacity`` dependency) because the behavior we need is small,
and we want full control over two things that matter for free-tier APIs:
exact backoff schedule, and an injectable ``sleep`` so tests run instantly.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    max_retries: int = 5,
    initial_backoff_s: float = 2.0,
    max_backoff_s: float = 60.0,
    jitter: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` and retry on the given exception types with exponential backoff.

    Args:
        fn: zero-arg callable to attempt.
        retry_on: exception types that should trigger a retry. Anything else
            propagates immediately (we only retry *transient* failures).
        max_retries: max retry attempts after the first try (so up to
            ``max_retries + 1`` total calls).
        initial_backoff_s: delay before the first retry; doubles each attempt.
        max_backoff_s: upper bound on any single delay.
        jitter: add up to 10% random jitter to avoid thundering-herd retries.
        sleep: sleep function, injectable so tests don't actually wait.

    Returns:
        Whatever ``fn`` returns on first success.

    Raises:
        The last exception if all retries are exhausted, or any exception not
        listed in ``retry_on`` immediately.
    """
    attempt = 0
    backoff = initial_backoff_s
    while True:
        try:
            return fn()
        except retry_on as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error("Giving up after %d retries: %s", max_retries, exc)
                raise
            delay = min(backoff, max_backoff_s)
            if jitter:
                delay += random.uniform(0, delay * 0.1)
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_retries,
                type(exc).__name__,
                delay,
            )
            sleep(delay)
            backoff = min(backoff * 2, max_backoff_s)
