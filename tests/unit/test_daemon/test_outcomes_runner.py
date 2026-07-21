"""Tests for the daily outcome-accrual tick (IBK-117)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from optionsbot.daemon.outcomes_runner import run_outcomes_tick


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.engine = MagicMock()
    ctx.ibkr = MagicMock()
    ctx.resolver = MagicMock()
    ctx.ibkr_lock = asyncio.Lock()  # real lock so `async with` works
    return ctx


async def test_run_outcomes_tick_evaluates() -> None:
    with patch(
        "optionsbot.daemon.outcomes_runner.evaluate_pending", new=AsyncMock(return_value=3)
    ) as ev, patch(
        "optionsbot.daemon.outcomes_runner.make_close_fetcher", return_value=AsyncMock()
    ):
        n = await run_outcomes_tick(
            _ctx(), now=datetime(2026, 7, 21, 20, 5, tzinfo=UTC)
        )
    assert n == 3 and ev.await_count == 1
    assert ev.await_args.args[2] == date(2026, 7, 22)


async def test_run_outcomes_tick_does_not_settle_zero_dte_before_close() -> None:
    with patch(
        "optionsbot.daemon.outcomes_runner.evaluate_pending", new=AsyncMock(return_value=0)
    ) as ev, patch(
        "optionsbot.daemon.outcomes_runner.make_close_fetcher", return_value=AsyncMock()
    ):
        await run_outcomes_tick(
            _ctx(), now=datetime(2026, 7, 21, 19, 59, tzinfo=UTC)
        )
    assert ev.await_args.args[2] == date(2026, 7, 21)
