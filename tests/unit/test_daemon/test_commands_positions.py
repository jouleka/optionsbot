"""Tests for the /positions Telegram command (IBK-112)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from optionsbot.config import Settings
from optionsbot.daemon.commands import dispatch


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.ibkr = MagicMock()
    ctx.resolver = MagicMock()
    ctx.ibkr_lock = asyncio.Lock()  # real lock so `async with` works
    return ctx


async def test_cmd_positions_renders_open_book() -> None:
    leg = {
        "sec_type": "OPT", "quantity": -1.0, "expiry": "20260717", "strike": 95.0,
        "right": "P", "market_price": 1.1, "unrealized_pnl": 30.0, "dte": 39, "delta": -0.28,
    }
    fake_view = {
        "net_unrealized_pnl": 30.0, "group_count": 1, "position_count": 1,
        "groups": [{"underlying": "SPY", "net_unrealized_pnl": 30.0, "legs": [leg]}],
    }
    with patch(
        "optionsbot.daemon.commands.assemble_open_book", new=AsyncMock(return_value=fake_view)
    ):
        replies = await dispatch(_ctx(), "/positions")
    assert len(replies) == 1
    assert "open book" in replies[0].text and "SPY" in replies[0].text


async def test_cmd_positions_ibkr_failure() -> None:
    with patch(
        "optionsbot.daemon.commands.assemble_open_book",
        new=AsyncMock(side_effect=ConnectionError("down")),
    ):
        replies = await dispatch(_ctx(), "/positions")
    assert "couldn't reach IBKR" in replies[0].text


async def test_cmd_positions_empty_book() -> None:
    empty = {"groups": [], "net_unrealized_pnl": 0.0, "group_count": 0}
    with patch(
        "optionsbot.daemon.commands.assemble_open_book", new=AsyncMock(return_value=empty)
    ):
        replies = await dispatch(_ctx(), "/positions")
    assert replies[0].text == "no open positions"


async def test_cmd_positions_renders_beta_footer() -> None:
    ctx = _ctx()
    ctx.settings = Settings()  # real settings: portfolio.enabled True, benchmark SPY
    view = {
        "net_unrealized_pnl": 0.0, "group_count": 1, "position_count": 1,
        "groups": [{"underlying": "SPY", "net_unrealized_pnl": 0.0, "legs": [
            {"sec_type": "OPT", "quantity": -1.0, "expiry": "20260717", "strike": 95.0,
             "right": "P", "market_price": 1.1, "unrealized_pnl": 0.0, "dte": 39,
             "delta": -0.3},
        ]}],
        "beta_weighted": {"dollar_per_1pct_spy": 480.0, "spy_equiv_shares": 80.0,
                          "underlyings_total": 1, "underlyings_covered": 1,
                          "complete": True, "benchmark": "SPY"},
    }
    mock = AsyncMock(return_value=view)
    with patch("optionsbot.daemon.commands.assemble_open_book", new=mock):
        replies = await dispatch(ctx, "/positions")
    assert "β-wtd" in replies[0].text and "SPY-eq" in replies[0].text
    # the command threads a HistoryClient + benchmark through to the orchestrator
    assert mock.await_args.kwargs["history_client"] is not None
    assert mock.await_args.kwargs["benchmark_symbol"] == "SPY"
