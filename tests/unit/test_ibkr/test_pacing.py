"""Tests for the rate-limiter primitives."""

import asyncio
import time

import pytest

from optionsbot.ibkr.pacing import ConcurrencyLimiter, RateLimiter


async def test_rate_limiter_allows_burst_below_limit() -> None:
    lim = RateLimiter(max_calls=5, window_seconds=10.0)
    start = time.monotonic()
    for _ in range(5):
        await lim.acquire()
    assert time.monotonic() - start < 0.05  # all 5 should be near-instant


async def test_rate_limiter_throttles_over_limit() -> None:
    # 3 calls per 0.2s window. Acquire 4 -> the 4th must wait ~0.2s.
    lim = RateLimiter(max_calls=3, window_seconds=0.2)
    for _ in range(3):
        await lim.acquire()
    start = time.monotonic()
    await lim.acquire()
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed <= 0.4, f"expected ~0.2s throttle, got {elapsed}"


async def test_rate_limiter_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        RateLimiter(max_calls=0, window_seconds=1.0)
    with pytest.raises(ValueError):
        RateLimiter(max_calls=1, window_seconds=0)


async def test_concurrency_limiter_caps_inflight() -> None:
    lim = ConcurrencyLimiter(max_concurrent=2)
    inflight = 0
    max_inflight_seen = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal inflight, max_inflight_seen
        async with lim:
            async with lock:
                inflight += 1
                max_inflight_seen = max(max_inflight_seen, inflight)
            await asyncio.sleep(0.05)
            async with lock:
                inflight -= 1

    await asyncio.gather(*(worker() for _ in range(8)))
    assert max_inflight_seen == 2


async def test_concurrency_limiter_rejects_bad_param() -> None:
    with pytest.raises(ValueError):
        ConcurrencyLimiter(max_concurrent=0)
