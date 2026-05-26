"""Async rate limiters used to respect IBKR TWS API pacing rules.

IBKR's documented historical-data limit is ~60 distinct requests per
10 minutes. Market-data (`reqMktData`) is limited by ticker line count
(account-tier dependent). We give callers two tools:

- ``RateLimiter`` -- a generic sliding-window rate limiter.
- ``ConcurrencyLimiter`` -- a thin wrapper over ``asyncio.Semaphore``
  for capping in-flight calls.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from types import TracebackType


class RateLimiter:
    """Sliding-window async rate limiter.

    ``await limiter.acquire()`` blocks until issuing one more call would
    keep the count of calls in the trailing ``window_seconds`` <= ``max_calls``.
    """

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._max = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                # Drop timestamps outside the trailing window.
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                # Sleep until the oldest timestamp slides out of the window.
                wait = self._window - (now - self._timestamps[0])
                await asyncio.sleep(wait)


class ConcurrencyLimiter:
    """Caps simultaneous in-flight calls. Use as an async context manager.

    Example::

        limiter = ConcurrencyLimiter(8)
        async with limiter:
            await client.some_call()
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self) -> ConcurrencyLimiter:
        await self._sem.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._sem.release()
