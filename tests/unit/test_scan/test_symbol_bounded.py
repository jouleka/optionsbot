"""_bounded_to_thread: blocking external calls run off-loop, time-bounded (IBK-149)."""

from __future__ import annotations

import time

from optionsbot.scan.symbol import _bounded_to_thread


async def test_bounded_to_thread_returns_value_when_fast() -> None:
    assert await _bounded_to_thread(lambda: "ok", timeout=1.0, default="x", label="t") == "ok"


async def test_bounded_to_thread_times_out_off_loop() -> None:
    """A hung blocking fn returns the default at ~timeout, NOT at the fn's
    duration -- proving the call ran off the event loop (a loop-blocking call
    couldn't honor the 0.2s timeout while sleeping 1.0s)."""
    def slow() -> str:
        time.sleep(1.0)
        return "real"

    start = time.monotonic()
    result = await _bounded_to_thread(slow, timeout=0.2, default="fallback", label="t")
    elapsed = time.monotonic() - start

    assert result == "fallback"
    assert elapsed < 0.7  # timeout fired well before slow's 1.0s -> not loop-blocked


async def test_bounded_to_thread_returns_default_on_error() -> None:
    def boom() -> str:
        raise ValueError("nope")

    assert await _bounded_to_thread(boom, timeout=1.0, default="fallback", label="t") == "fallback"
