"""Tests for the open-book position view (IBK-112)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from optionsbot.analysis.positions import (
    assemble_open_book,
    build_positions_view,
    position_dte,
)
from optionsbot.ibkr.types import OptionQuote, PortfolioPosition

_TODAY = datetime(2026, 6, 8, tzinfo=UTC)


def _pp(
    symbol: str, sec_type: str = "OPT", expiry: str = "20260717", strike: float = 95.0,
    right: str = "P", position: float = -1.0, upnl: float = 45.0, market_price: float = 1.1,
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol=symbol, sec_type=sec_type,
        expiry=expiry if sec_type == "OPT" else None,
        strike=strike if sec_type == "OPT" else None,
        right=right if sec_type == "OPT" else None,  # type: ignore[arg-type]
        multiplier=100 if sec_type == "OPT" else 1, position=position, avg_cost=250.0,
        market_price=market_price, market_value=-110.0, unrealized_pnl=upnl, realized_pnl=0.0,
    )


def _q(strike: float = 95.0, delta: float = -0.28) -> OptionQuote:
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right="P", bid=1.0, ask=1.2,
        last=1.1, mid=1.1, iv=0.22, delta=delta, gamma=0.01, theta=-0.05, vega=0.1,
        open_interest=10, volume=5, ts=_TODAY, delayed=True,
    )


def test_position_dte() -> None:
    assert position_dte("20260717", _TODAY.date()) == 39
    assert position_dte(None, _TODAY.date()) is None
    assert position_dte("not-a-date", _TODAY.date()) is None


def test_build_view_groups_and_sums() -> None:
    positions = [
        _pp("SPY", strike=95.0, position=-1.0, upnl=45.0),
        _pp("SPY", strike=90.0, position=1.0, upnl=-15.0),
        _pp("AAPL", sec_type="STK", position=100.0, upnl=500.0),
    ]
    greeks = {("SPY", "20260717", 95.0, "P"): _q()}
    view = build_positions_view(positions, greeks, _TODAY)
    assert view["net_unrealized_pnl"] == 530.0
    assert view["group_count"] == 2 and view["position_count"] == 3
    spy = next(g for g in view["groups"] if g["underlying"] == "SPY")
    assert spy["net_unrealized_pnl"] == 30.0
    assert [lg["strike"] for lg in spy["legs"]] == [90.0, 95.0]  # sorted (dte, strike)
    leg95 = next(lg for lg in spy["legs"] if lg["strike"] == 95.0)
    assert leg95["dte"] == 39 and leg95["delta"] == -0.28
    leg90 = next(lg for lg in spy["legs"] if lg["strike"] == 90.0)
    assert leg90["delta"] is None  # no greeks entry -> None
    aapl = next(g for g in view["groups"] if g["underlying"] == "AAPL")
    assert aapl["legs"][0]["sec_type"] == "STK" and aapl["legs"][0]["dte"] is None


def test_build_view_empty() -> None:
    view = build_positions_view([], {}, _TODAY)
    assert view["groups"] == [] and view["net_unrealized_pnl"] == 0.0
    assert view["group_count"] == 0 and view["position_count"] == 0


async def test_assemble_open_book_tolerates_greeks_failure() -> None:
    pos_client = AsyncMock()
    pos_client.get_portfolio.return_value = [
        _pp("SPY", strike=95.0, position=-1.0, upnl=45.0),
        _pp("SPY", strike=90.0, position=1.0, upnl=-15.0),
    ]
    md_client = AsyncMock()
    # 95P returns greeks; 90P raises -> tolerated (leg keeps P&L, no greeks).
    md_client.get_option_snapshot.side_effect = [
        _q(strike=95.0, delta=-0.28),
        RuntimeError("no data"),
    ]
    view = await assemble_open_book(pos_client, md_client, _TODAY)
    spy = view["groups"][0]
    leg95 = next(lg for lg in spy["legs"] if lg["strike"] == 95.0)
    leg90 = next(lg for lg in spy["legs"] if lg["strike"] == 90.0)
    assert leg95["delta"] == -0.28
    assert leg90["delta"] is None and leg90["unrealized_pnl"] == -15.0
