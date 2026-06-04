"""build_suggestion derives expected_value from the realized-vol payoff model (IBK-103)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from optionsbot.analysis.types import MarketView
from optionsbot.strategies.base import Leg, StrategySnapshot
from optionsbot.strategies.iron_condor import IronCondor


def _expiry(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).strftime("%Y%m%d")


def _snapshot(hv20: float | None) -> StrategySnapshot:
    view = MarketView(
        direction="neutral", direction_strength="weak", iv_regime="high",
        iv_rank_value=None, earnings_in_window=False, warming_up=True,
    )
    return StrategySnapshot(
        symbol="SPY", spot=100.0, atm_iv=0.30, hv20=hv20, iv_rank=None,
        chain=(), view=view, dte_target=45,
    )


def test_estimate_expected_value_finite_with_hv() -> None:
    exp = _expiry(30)
    legs = (Leg(symbol="SPY", side="sell", sec_type="OPT", expiry=exp, strike=90.0, right="P"),)
    ev = IronCondor().estimate_expected_value(legs, _snapshot(hv20=0.18))
    assert ev is not None and math.isfinite(ev)


def test_estimate_expected_value_none_without_hv() -> None:
    exp = _expiry(30)
    legs = (Leg(symbol="SPY", side="sell", sec_type="OPT", expiry=exp, strike=90.0, right="P"),)
    assert IronCondor().estimate_expected_value(legs, _snapshot(hv20=None)) is None


def test_estimate_expected_value_none_for_multi_expiry() -> None:
    legs = (
        Leg(symbol="SPY", side="sell", sec_type="OPT", expiry=_expiry(30), strike=100.0, right="C"),
        Leg(symbol="SPY", side="buy", sec_type="OPT", expiry=_expiry(60), strike=100.0, right="C"),
    )
    assert IronCondor().estimate_expected_value(legs, _snapshot(hv20=0.18)) is None
