"""The base class now derives prob_profit from the payoff module (IBK-93)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from optionsbot.analysis.types import MarketView
from optionsbot.strategies.base import Leg, StrategySnapshot
from optionsbot.strategies.iron_condor import IronCondor


def _expiry(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).strftime("%Y%m%d")


def _snapshot(atm_iv: float | None) -> StrategySnapshot:
    view = MarketView(
        direction="neutral", direction_strength="weak", iv_regime="high",
        iv_rank_value=None, earnings_in_window=False, warming_up=True,
    )
    return StrategySnapshot(
        symbol="SPY", spot=100.0, atm_iv=atm_iv, hv20=0.18, iv_rank=None,
        chain=(), view=view, dte_target=45,
    )


def test_estimate_prob_profit_reflects_legs_not_a_constant() -> None:
    # An ITM long call has a higher P(profit) than a far-OTM one. The old
    # iron_condor constant ignored the legs (same value for both), so this
    # comparison is the red->green proof the model now uses the strikes.
    # (Empty chain -> estimate_credit ~ 0; exact probabilities are covered in
    # test_payoff.py, so here we only assert the relative ordering + range.)
    exp = _expiry(30)
    snap = _snapshot(atm_iv=0.20)
    otm = (Leg(symbol="SPY", side="buy", sec_type="OPT", expiry=exp, strike=120.0, right="C"),)
    itm = (Leg(symbol="SPY", side="buy", sec_type="OPT", expiry=exp, strike=95.0, right="C"),)
    p_otm = IronCondor().estimate_prob_profit(otm, snap)
    p_itm = IronCondor().estimate_prob_profit(itm, snap)
    assert p_otm is not None and p_itm is not None
    assert 0.0 <= p_otm <= 1.0 and 0.0 <= p_itm <= 1.0
    assert p_itm > p_otm


def test_estimate_prob_profit_none_without_iv() -> None:
    exp = _expiry(30)
    legs = (Leg(symbol="SPY", side="buy", sec_type="OPT", expiry=exp, strike=100.0, right="C"),)
    assert IronCondor().estimate_prob_profit(legs, _snapshot(atm_iv=None)) is None
