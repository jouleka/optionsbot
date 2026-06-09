"""Tests for the plain-text open-book formatter (IBK-112)."""

from __future__ import annotations

from typing import Any

from optionsbot.alerts.formatter import format_positions_text


def _view() -> dict[str, Any]:
    return {
        "net_unrealized_pnl": 30.0, "group_count": 1, "position_count": 2,
        "groups": [{"underlying": "SPY", "net_unrealized_pnl": 30.0, "legs": [
            {"sec_type": "OPT", "quantity": -1.0, "expiry": "20260717", "strike": 95.0,
             "right": "P", "market_price": 1.1, "unrealized_pnl": 45.0, "dte": 39, "delta": -0.28},
            {"sec_type": "STK", "quantity": 100.0, "expiry": None, "strike": None, "right": None,
             "market_price": 185.0, "unrealized_pnl": -15.0, "dte": None, "delta": None},
        ]}],
    }


def test_format_positions_text_renders_groups_and_legs() -> None:
    out = format_positions_text(_view())
    assert "open book" in out and "SPY" in out
    assert "95P" in out and "17Jul" in out and "DTE 39" in out
    assert "shares" in out  # stock leg rendered


def test_format_positions_text_empty() -> None:
    out = format_positions_text({"groups": [], "net_unrealized_pnl": 0.0, "group_count": 0})
    assert out == "no open positions"


def test_format_positions_text_option_leg_missing_greeks() -> None:
    # The likely live path: best-effort Greeks fetch failed -> OPT leg with None
    # delta/mid/P&L/DTE must render safely (IBK-112 review).
    leg = {
        "sec_type": "OPT", "quantity": -1.0, "expiry": "20260717", "strike": 95.0,
        "right": "P", "market_price": None, "unrealized_pnl": None, "dte": None, "delta": None,
    }
    view = {
        "net_unrealized_pnl": 0.0, "group_count": 1, "position_count": 1,
        "groups": [{"underlying": "SPY", "net_unrealized_pnl": 0.0, "legs": [leg]}],
    }
    out = format_positions_text(view)
    assert "mid ?" in out and "DTE ?" in out and "P&L $?" in out
    assert "Δ" not in out  # no delta rendered when missing (this leg has no Greeks footer)


def _view_with_greeks(complete: bool = True) -> dict[str, Any]:
    return {
        "net_unrealized_pnl": 30.0, "group_count": 1, "position_count": 1,
        "groups": [{"underlying": "SPY", "net_unrealized_pnl": 30.0, "legs": [
            {"sec_type": "OPT", "quantity": -1.0, "expiry": "20260717", "strike": 95.0,
             "right": "P", "market_price": 1.1, "unrealized_pnl": 30.0, "dte": 39, "delta": -0.3},
        ]}],
        "portfolio_greeks": {
            "net_delta": -250.0, "net_gamma": -3.2, "net_theta": 45.0, "net_vega": -120.0,
            "option_legs_total": 5, "option_legs_with_greeks": 5 if complete else 4,
            "complete": complete,
        },
    }


def test_format_positions_text_has_greeks_footer() -> None:
    out = format_positions_text(_view_with_greeks())
    assert "book greeks" in out and "-250" in out and "/day" in out
    assert "legs)" not in out  # complete -> no coverage note


def test_format_positions_text_greeks_partial_note() -> None:
    out = format_positions_text(_view_with_greeks(complete=False))
    assert "(4/5 legs)" in out


def _view_with_beta(
    *, complete: bool = True, benchmark_spot: bool = True, covered: int = 2
) -> dict[str, Any]:
    v = _view_with_greeks()
    v["beta_weighted"] = {
        "beta_weighted_dollar_delta": 48000.0,
        "dollar_per_1pct_spy": 480.0,
        "spy_equiv_shares": 80.0 if benchmark_spot else None,
        "underlyings_total": 2,
        "underlyings_covered": covered,
        "complete": complete,
        "benchmark": "SPY",
    }
    return v


def test_format_positions_text_beta_footer_complete() -> None:
    out = format_positions_text(_view_with_beta())
    assert "β-wtd" in out and "+80 SPY-eq" in out and "/1% SPY" in out
    assert "underlyings)" not in out


def test_format_positions_text_beta_footer_partial_coverage() -> None:
    out = format_positions_text(_view_with_beta(complete=False, covered=1))
    assert "(1/2 underlyings)" in out


def test_format_positions_text_beta_footer_no_benchmark_spot() -> None:
    out = format_positions_text(_view_with_beta(benchmark_spot=False))
    assert "SPY-eq" not in out and "/1% SPY" in out


def test_format_positions_text_beta_footer_zero_coverage() -> None:
    out = format_positions_text(_view_with_beta(complete=False, covered=0))
    assert "β-wtd: n/a" in out


def test_format_positions_text_beta_footer_no_weightable_positions() -> None:
    # Fully delta-neutral book: nothing to weight. Must say so, not print "+0 SPY-eq"
    # as if it had measured a flat book (honesty wart from Opus review S2).
    v = _view_with_greeks()
    v["beta_weighted"] = {
        "beta_weighted_dollar_delta": 0.0, "dollar_per_1pct_spy": 0.0,
        "spy_equiv_shares": 0.0, "underlyings_total": 0, "underlyings_covered": 0,
        "complete": True, "benchmark": "SPY",
    }
    out = format_positions_text(v)
    assert "β-wtd: n/a" in out and "SPY-eq" not in out


def test_format_positions_text_beta_footer_absent_when_none() -> None:
    v = _view_with_greeks()
    v["beta_weighted"] = None
    out = format_positions_text(v)
    assert "β-wtd" not in out
