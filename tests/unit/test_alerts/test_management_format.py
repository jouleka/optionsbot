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


def _p(
    trigger: str, net_pnl: float = 50.0, base_amount: float = 80.0, basis: str = "credit"
) -> ProfitAlert:
    return ProfitAlert(
        symbol="SPY", trigger=trigger, basis=basis, base_amount=base_amount, net_pnl=net_pnl,
        profit_pct=net_pnl / base_amount, dedup_key=f"SPY:profit:{trigger}",
    )


def test_format_take_profit() -> None:
    out = format_profit_alert(_p("take_profit", 50.0, 80.0))
    assert "take profit" in out and "SPY" in out and "62%" in out and "$80 credit" in out


def test_format_stop_loss() -> None:
    out = format_profit_alert(_p("stop_loss", -170.0, 80.0))
    assert "stop loss" in out and "SPY" in out and "2.1x" in out and "credit" in out


def test_format_debit_take_profit() -> None:
    out = format_profit_alert(_p("take_profit", 200.0, 250.0, basis="debit"))
    assert "take profit" in out and "+80%" in out and "$250 debit" in out


def test_format_debit_stop() -> None:
    out = format_profit_alert(_p("stop_loss", -125.0, 250.0, basis="debit"))
    assert "stop loss" in out and "-50%" in out and "$250 debit" in out
