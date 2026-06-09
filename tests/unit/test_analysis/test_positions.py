"""Tests for the open-book position view (IBK-112)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pandas as pd

from optionsbot.analysis.positions import (
    assemble_open_book,
    build_positions_view,
    per_underlying_share_delta,
    portfolio_greeks,
    position_dte,
)
from optionsbot.ibkr.types import OptionQuote, PortfolioPosition, StockQuote

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


# --- portfolio_greeks (IBK-115) --------------------------------------------


def _oq(
    delta: float | None = None, gamma: float | None = None,
    theta: float | None = None, vega: float | None = None,
    strike: float = 95.0, right: str = "P",
) -> OptionQuote:
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right=right, bid=1.0, ask=1.2,
        last=1.1, mid=1.1, iv=0.22, delta=delta, gamma=gamma, theta=theta, vega=vega,
        open_interest=10, volume=5, ts=_TODAY, delayed=True,
    )


def test_portfolio_greeks_sign_conventions() -> None:
    # short put: delta -0.30, pos -1 -> +30 delta; +theta, -vega, -gamma.
    sp = _pp("SPY", strike=95.0, right="P", position=-1.0)
    greeks = {
        ("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30, gamma=0.01, theta=-0.05, vega=0.10),
    }
    g = portfolio_greeks([sp], greeks)
    assert g["net_delta"] == 30.0          # -0.30 * -1 * 100
    assert g["net_theta"] == 5.0           # -0.05 * -1 * 100 -> collect theta
    assert g["net_vega"] == -10.0          # 0.10 * -1 * 100 -> short vega
    assert round(g["net_gamma"], 4) == -1.0
    assert g["option_legs_total"] == 1 and g["option_legs_with_greeks"] == 1
    assert g["complete"] is True


def test_portfolio_greeks_short_call_negative_delta() -> None:
    sc = _pp("SPY", strike=105.0, right="C", position=-1.0)
    greeks = {("SPY", "20260717", 105.0, "C"): _oq(delta=0.30, strike=105.0, right="C")}
    assert portfolio_greeks([sc], greeks)["net_delta"] == -30.0


def test_portfolio_greeks_includes_stock_delta_and_multiplier() -> None:
    stock = _pp("SPY", sec_type="STK", position=100.0)            # +100 delta (1/share)
    longput2 = _pp("SPY", strike=95.0, right="P", position=2.0)   # long 2 -> x2 scaling
    greeks = {("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30)}
    g = portfolio_greeks([stock, longput2], greeks)
    assert g["net_delta"] == 100.0 + (-0.30 * 2 * 100)           # 100 - 60 = 40
    assert g["option_legs_total"] == 1 and g["option_legs_with_greeks"] == 1


def test_portfolio_greeks_partial_coverage() -> None:
    a = _pp("SPY", strike=95.0, right="P", position=-1.0)
    b = _pp("SPY", strike=90.0, right="P", position=1.0)         # no greeks entry
    greeks = {("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30)}
    g = portfolio_greeks([a, b], greeks)
    assert g["option_legs_total"] == 2 and g["option_legs_with_greeks"] == 1
    assert g["complete"] is False
    assert g["net_delta"] == 30.0                                # only the covered leg


def test_portfolio_greeks_empty() -> None:
    g = portfolio_greeks([], {})
    assert g["net_delta"] == 0.0 and g["complete"] is True
    assert g["option_legs_total"] == 0 and g["option_legs_with_greeks"] == 0


def test_build_positions_view_includes_portfolio_greeks() -> None:
    sp = _pp("SPY", strike=95.0, right="P", position=-1.0)
    greeks = {("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30)}
    view = build_positions_view([sp], greeks, _TODAY)
    assert "portfolio_greeks" in view and view["portfolio_greeks"]["net_delta"] == 30.0


def test_portfolio_greeks_delta_present_theta_none() -> None:
    # A leg with delta but no theta/vega counts as covered for delta, but contributes
    # nothing to net_theta/net_vega (IBK-115 review S1/S2).
    sp = _pp("SPY", strike=95.0, right="P", position=-1.0)
    greeks = {("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30, theta=None, vega=None)}
    g = portfolio_greeks([sp], greeks)
    assert g["net_delta"] == 30.0 and g["net_theta"] == 0.0 and g["net_vega"] == 0.0
    assert g["option_legs_with_greeks"] == 1 and g["complete"] is True


def test_portfolio_greeks_cross_underlying_delta_is_unit_mixed() -> None:
    # DOCUMENTED LIMITATION: delta sums across underlyings in raw share-equiv, so long 100
    # SPY shares + a short AAPL call (delta 1.0, pos -1 -> -100) net to 0 even though the
    # book is not actually market-neutral. Pins the known caveat (IBK-115 review S3).
    spy = _pp("SPY", sec_type="STK", position=100.0)
    aapl = _pp("AAPL", strike=150.0, right="C", position=-1.0)
    greeks = {("AAPL", "20260717", 150.0, "C"): _oq(delta=1.0, strike=150.0, right="C")}
    assert portfolio_greeks([spy, aapl], greeks)["net_delta"] == 0.0


# --- per_underlying_share_delta (IBK-118) ----------------------------------


def test_per_underlying_share_delta_groups_and_scales() -> None:
    spy_put = _pp("SPY", strike=95.0, right="P", position=-1.0)  # -0.30 * -1 * 100 = +30
    spy_stock = _pp("SPY", sec_type="STK", position=100.0)  # +100
    aapl = _pp("AAPL", strike=150.0, right="C", position=-1.0)  # 1.0 * -1 * 100 = -100
    greeks = {
        ("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30),
        ("AAPL", "20260717", 150.0, "C"): _oq(delta=1.0, strike=150.0, right="C"),
    }
    out = per_underlying_share_delta([spy_put, spy_stock, aapl], greeks)
    assert out["SPY"] == 130.0  # 30 + 100
    assert out["AAPL"] == -100.0


def test_per_underlying_share_delta_excludes_missing_greeks() -> None:
    a = _pp("SPY", strike=95.0, right="P", position=-1.0)
    b = _pp("SPY", strike=90.0, right="P", position=1.0)  # no greeks entry -> excluded
    greeks = {("SPY", "20260717", 95.0, "P"): _oq(delta=-0.30)}
    out = per_underlying_share_delta([a, b], greeks)
    assert out["SPY"] == 30.0  # only the leg with greeks contributes


# --- assemble_open_book beta-weighting enrichment (IBK-118) ----------------

_BENCH_RETS_OB = [0.01, -0.012, 0.008, -0.005, 0.015, -0.009] * 10  # 60 returns


def _closes_ob(returns: list[float], start: float = 100.0) -> pd.DataFrame:
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    return pd.DataFrame({"close": closes})


def _sq(last: float) -> StockQuote:
    return StockQuote(
        symbol="X", bid=last, ask=last, last=last, mid=last, ts=_TODAY, delayed=True
    )


def test_build_view_labels_structure() -> None:
    # A bull put spread on SPY -> group carries the structure label.
    legs = [
        _pp("SPY", strike=95.0, right="P", position=-1.0, upnl=20.0),
        _pp("SPY", strike=90.0, right="P", position=1.0, upnl=-5.0),
    ]
    view = build_positions_view(legs, {}, _TODAY)
    spy = next(g for g in view["groups"] if g["underlying"] == "SPY")
    assert spy["structure"] == "Bull Put Spread"


def test_build_view_custom_structure() -> None:
    legs = [
        _pp("SPY", strike=95.0, right="P", position=-1.0),
        _pp("SPY", strike=90.0, right="P", position=1.0),
        _pp("SPY", strike=105.0, right="C", position=-1.0),
    ]
    view = build_positions_view(legs, {}, _TODAY)
    spy = next(g for g in view["groups"] if g["underlying"] == "SPY")
    assert spy["structure"] == "custom (3 legs)"


async def test_assemble_open_book_no_history_client_has_no_beta() -> None:  # regression
    pos_client = AsyncMock()
    pos_client.get_portfolio.return_value = [_pp("SPY", strike=95.0, position=-1.0)]
    md_client = AsyncMock()
    md_client.get_option_snapshot.return_value = _q(strike=95.0, delta=-0.28)
    view = await assemble_open_book(pos_client, md_client, _TODAY)
    assert "beta_weighted" not in view


async def test_assemble_open_book_beta_weights_when_history_provided() -> None:
    spy_put = _pp("SPY", strike=95.0, right="P", position=-1.0)  # +30 share-delta
    aapl_stock = _pp("AAPL", sec_type="STK", position=100.0)  # +100 share-delta
    pos_client = AsyncMock()
    pos_client.get_portfolio.return_value = [spy_put, aapl_stock]
    md_client = AsyncMock()
    md_client.get_option_snapshot.return_value = _oq(delta=-0.30)  # SPY 30 share-delta
    md_client.get_stock_snapshot.side_effect = lambda symbol: _sq(
        {"SPY": 600.0, "AAPL": 200.0}[symbol]
    )
    spy_bars = _closes_ob(_BENCH_RETS_OB)
    aapl_bars = _closes_ob([1.5 * r for r in _BENCH_RETS_OB])  # beta 1.5 vs SPY
    history_client = AsyncMock()
    history_client.get_history.side_effect = lambda symbol, days=0: {
        "SPY": spy_bars,
        "AAPL": aapl_bars,
    }[symbol]
    view = await assemble_open_book(
        pos_client,
        md_client,
        _TODAY,
        history_client=history_client,
        benchmark_symbol="SPY",
        beta_window=60,
    )
    bw = view["beta_weighted"]
    # S = 1.0*30*600 + 1.5*100*200 = 18000 + 30000 = 48000
    assert round(bw["beta_weighted_dollar_delta"]) == 48000
    assert round(bw["spy_equiv_shares"]) == 80  # 48000 / 600
    assert bw["underlyings_covered"] == 2 and bw["complete"] is True
    assert bw["benchmark"] == "SPY"


async def test_assemble_open_book_benchmark_history_failure_yields_none() -> None:
    pos_client = AsyncMock()
    pos_client.get_portfolio.return_value = [_pp("SPY", strike=95.0, position=-1.0)]
    md_client = AsyncMock()
    md_client.get_option_snapshot.return_value = _oq(delta=-0.30)
    history_client = AsyncMock()
    history_client.get_history.side_effect = RuntimeError("no data")
    view = await assemble_open_book(
        pos_client,
        md_client,
        _TODAY,
        history_client=history_client,
        benchmark_symbol="SPY",
        beta_window=60,
    )
    assert view["beta_weighted"] is None


async def test_assemble_open_book_per_underlying_failure_dings_coverage() -> None:
    spy_put = _pp("SPY", strike=95.0, right="P", position=-1.0)  # benchmark, beta=1
    aapl_stock = _pp("AAPL", sec_type="STK", position=100.0)
    pos_client = AsyncMock()
    pos_client.get_portfolio.return_value = [spy_put, aapl_stock]
    md_client = AsyncMock()
    md_client.get_option_snapshot.return_value = _oq(delta=-0.30)
    md_client.get_stock_snapshot.side_effect = lambda symbol: _sq(600.0)
    spy_bars = _closes_ob(_BENCH_RETS_OB)

    def _hist(symbol: str, days: int = 0) -> pd.DataFrame:
        if symbol == "AAPL":
            raise RuntimeError("no AAPL history")
        return spy_bars

    history_client = AsyncMock()
    history_client.get_history.side_effect = _hist
    view = await assemble_open_book(
        pos_client,
        md_client,
        _TODAY,
        history_client=history_client,
        benchmark_symbol="SPY",
        beta_window=60,
    )
    bw = view["beta_weighted"]
    # SPY covered (beta=1 short-circuit); AAPL history failed -> excluded.
    assert bw["underlyings_total"] == 2 and bw["underlyings_covered"] == 1
    assert bw["complete"] is False
