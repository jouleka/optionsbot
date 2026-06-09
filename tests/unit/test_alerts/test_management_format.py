"""Tests for the plain-text management-alert formatter (IBK-113)."""

from __future__ import annotations

from optionsbot.alerts.formatter import format_management_alert, format_profit_alert
from optionsbot.analysis.management import ManagementAlert, ProfitAlert


def _a(
    triggers: tuple[str, ...], dte: int | None = 7, spot: float | None = 93.2,
    right: str = "P", quantity: float = -1.0, itm: bool | None = None,
) -> ManagementAlert:
    return ManagementAlert(
        symbol="SPY", expiry="20260717", strike=95.0, right=right, quantity=quantity,
        triggers=triggers, dte=dte, spot=spot, itm=itm,
        dedup_key="SPY:20260717:95:" + right + ":" + "+".join(triggers),
    )


def test_format_assignment_only_put() -> None:
    out = format_management_alert(_a(("assignment",), dte=40, spot=93.2, itm=True))
    assert "assignment risk" in out and "95P" in out and "93.20" in out
    assert "short put ITM" in out and "40 DTE" in out


def test_format_assignment_only_call() -> None:
    out = format_management_alert(_a(("assignment",), dte=40, spot=97.0, right="C", itm=True))
    assert "short call ITM" in out and "97.00 > 95" in out


def test_format_dte_manage_short() -> None:
    out = format_management_alert(_a(("dte_manage",), dte=18, spot=None))
    assert "manage" in out and "18 DTE" in out and "short option approaching expiry" in out


def test_format_dte_urgent_short() -> None:
    out = format_management_alert(_a(("dte_urgent",), dte=5, spot=None))
    assert "URGENT" in out and "5 DTE" in out


def test_format_short_merged_one_message() -> None:
    out = format_management_alert(_a(("assignment", "dte_urgent"), dte=3, spot=92.0, itm=True))
    assert "URGENT" in out and "short put ITM" in out and "92.00 < 95" in out
    assert "approaching expiry" in out
    assert out.count("\n") == 0  # a single line / message


def test_format_long_itm() -> None:
    out = format_management_alert(
        _a(("dte_urgent",), dte=3, spot=155.0, right="C", quantity=2.0, itm=True)
    )
    assert "long call ITM" in out and "auto-exercises" in out


def test_format_long_otm() -> None:
    out = format_management_alert(
        _a(("dte_manage",), dte=18, spot=99.0, quantity=1.0, itm=False)
    )
    assert "long option approaching expiry" in out and "premium decaying" in out


def test_format_long_spot_unknown() -> None:
    out = format_management_alert(_a(("dte_urgent",), dte=3, spot=None, quantity=1.0, itm=None))
    assert "long option approaching expiry" in out and "ITM" not in out


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
