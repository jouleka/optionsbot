"""Tests for position-management triggers (IBK-113)."""

from __future__ import annotations

from datetime import date

from optionsbot.analysis.management import evaluate_position_triggers
from optionsbot.config import ManageSettings
from optionsbot.ibkr.types import PortfolioPosition

_TODAY = date(2026, 6, 8)


def _pp(
    strike: float = 95.0, right: str = "P", position: float = -1.0,
    expiry: str = "20260626", sec_type: str = "OPT",
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol="SPY", sec_type=sec_type,
        expiry=expiry if sec_type == "OPT" else None,
        strike=strike if sec_type == "OPT" else None,
        right=right if sec_type == "OPT" else None,  # type: ignore[arg-type]
        multiplier=100 if sec_type == "OPT" else 1, position=position, avg_cost=250.0,
        market_price=1.0, market_value=-100.0, unrealized_pnl=10.0, realized_pnl=0.0,
    )


def _settings() -> ManageSettings:
    return ManageSettings()  # manage_dte=21, urgent_dte=7


def test_dte_manage_bucket() -> None:
    out = evaluate_position_triggers([_pp(expiry="20260626")], {}, _TODAY, _settings())  # 18 DTE
    assert len(out) == 1 and out[0].trigger == "dte_manage" and out[0].dte == 18


def test_dte_urgent_bucket_only() -> None:
    out = evaluate_position_triggers([_pp(expiry="20260613")], {}, _TODAY, _settings())  # 5 DTE
    assert [a.trigger for a in out] == ["dte_urgent"]


def test_no_alert_when_far_dated() -> None:
    assert evaluate_position_triggers([_pp(expiry="20260717")], {}, _TODAY, _settings()) == []


def test_assignment_short_put_itm() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260717")], {"SPY": 93.2}, _TODAY, _settings()
    )
    assert [a.trigger for a in out] == ["assignment"]
    assert out[0].spot == 93.2 and out[0].dedup_key == "SPY:20260717:95:P:assignment"


def test_assignment_short_call_itm() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="C", expiry="20260717")], {"SPY": 97.0}, _TODAY, _settings()
    )
    assert [a.trigger for a in out] == ["assignment"]


def test_otm_short_no_assignment() -> None:
    assert evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260717")], {"SPY": 99.0}, _TODAY, _settings()
    ) == []


def test_long_leg_ignored() -> None:
    assert evaluate_position_triggers(
        [_pp(position=1.0, expiry="20260613")], {"SPY": 90.0}, _TODAY, _settings()
    ) == []


def test_missing_spot_skips_assignment_keeps_dte() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260613")], {}, _TODAY, _settings()
    )
    assert [a.trigger for a in out] == ["dte_urgent"]


def test_assignment_disabled_by_settings() -> None:
    s = ManageSettings(assignment_alerts=False)
    assert evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260717")], {"SPY": 90.0}, _TODAY, s
    ) == []


def test_stock_leg_ignored() -> None:
    assert evaluate_position_triggers(
        [_pp(sec_type="STK", position=-100.0)], {"SPY": 90.0}, _TODAY, _settings()
    ) == []
