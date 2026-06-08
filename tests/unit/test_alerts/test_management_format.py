"""Tests for the plain-text management-alert formatter (IBK-113)."""

from __future__ import annotations

from optionsbot.alerts.formatter import format_management_alert
from optionsbot.analysis.management import ManagementAlert


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
