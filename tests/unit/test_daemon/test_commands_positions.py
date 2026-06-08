"""Tests for the /positions Telegram command (IBK-112)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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
