"""Tests for the plain-text management-alert formatter (IBK-113)."""

from __future__ import annotations

from optionsbot.alerts.formatter import format_management_alert, format_profit_alert
from optionsbot.analysis.management import ManagementAlert, ProfitAlert


def _a(
    trigger: str, dte: int | None = 7, spot: float | None = 93.2, right: str = "P"
) -> ManagementAlert:
    return ManagementAlert(
        symbol="SPY", expiry="20260717", strike=95.0, right=right, quantity=-1.0,
        trigger=trigger, dte=dte, spot=spot, dedup_key=f"SPY:20260717:95:{right}:{trigger}",
    )


def test_format_assignment_put() -> None:
    out = format_management_alert(_a("assignment"))
    assert "assignment risk" in out and "SPY" in out and "95P" in out and "93.20" in out
    assert "short put ITM" in out


def test_format_assignment_call() -> None:
    out = format_management_alert(_a("assignment", spot=97.0, right="C"))
    assert "short call ITM" in out and "97.00 > 95" in out


def test_format_dte_manage() -> None:
    out = format_management_alert(_a("dte_manage", dte=18, spot=None))
    assert "manage" in out and "18 DTE" in out


def test_format_dte_urgent() -> None:
    out = format_management_alert(_a("dte_urgent", dte=5, spot=None))
    assert "URGENT" in out and "5 DTE" in out


def _p(trigger: str, net_credit: float = 80.0, net_pnl: float = 50.0) -> ProfitAlert:
    return ProfitAlert(
        symbol="SPY", trigger=trigger, net_credit=net_credit, net_pnl=net_pnl,
        profit_pct=net_pnl / net_credit, dedup_key=f"SPY:profit:{trigger}",
    )


def test_format_take_profit() -> None:
    out = format_profit_alert(_p("take_profit", 80.0, 50.0))
    assert "take profit" in out and "SPY" in out and "62%" in out and "$80" in out


def test_format_stop_loss() -> None:
    out = format_profit_alert(_p("stop_loss", 80.0, -170.0))
    assert "stop loss" in out and "SPY" in out and "2.1x" in out
