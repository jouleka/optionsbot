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
    assert "Δ" not in out  # no delta rendered when missing
