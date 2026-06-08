"""base.py wiring: smile-aware PoP/EV with graceful fallback to flat (IBK-111)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.scoring.payoff import expected_value_dollars, prob_of_profit
from optionsbot.strategies.base import Leg, StrategySnapshot
from optionsbot.strategies.verticals import BullPutSpread

# Skewed put wing (richer OTM puts) + flat calls. Keyed by strike.
_PUT_IV = {80.0: 0.40, 90.0: 0.30, 95.0: 0.25, 100.0: 0.20}
_CALL_IV = {100.0: 0.20, 110.0: 0.20, 120.0: 0.20}


def _expiry(days: int) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y%m%d")


def _leg(right: str, strike: float, iv: float | None, exp: str) -> OptionChainLeg:
    # Premium increases with strike so the short (higher-strike) put nets a credit.
    mid = max(0.5, (strike - 70.0) * 0.1)
    return OptionChainLeg(
        symbol="SPY", expiry=exp, strike=strike, right=right, bid=mid - 0.05, ask=mid + 0.05,
        iv=iv, delta=(-0.30 if right == "P" else 0.30), gamma=None, theta=None,
        vega=None, open_interest=10, volume=10,
    )


def _chain(exp: str, *, with_iv: bool) -> tuple[OptionChainLeg, ...]:
    puts = tuple(_leg("P", k, (_PUT_IV[k] if with_iv else None), exp) for k in _PUT_IV)
    calls = tuple(_leg("C", k, (_CALL_IV[k] if with_iv else None), exp) for k in _CALL_IV)
    return puts + calls


def _snapshot(chain: tuple[OptionChainLeg, ...]) -> StrategySnapshot:
    return StrategySnapshot(
        symbol="SPY", spot=100.0, atm_iv=0.20, hv20=0.18, iv_rank=0.5,
        chain=chain, view=MarketView("bull", "weak", "high", 0.5, False, False), dte_target=30,
    )


def _legs(exp: str) -> tuple[Leg, ...]:
    return (Leg("SPY", "sell", "OPT", exp, 95.0, "P"), Leg("SPY", "buy", "OPT", exp, 90.0, "P"))


def _dte(exp: str) -> float:
    return float((datetime.strptime(exp, "%Y%m%d").date() - date.today()).days)


def test_estimate_prob_profit_uses_smile_and_lowers_pop() -> None:
    exp = _expiry(30)
    legs, snap = _legs(exp), _snapshot(_chain(_expiry(30), with_iv=True))
    strat = BullPutSpread()
    pop = strat.estimate_prob_profit(legs, snap)
    # Reference: the flat ATM-IV lognormal the un-wired path would return.
    credit = strat.estimate_credit(legs, snap)
    flat_ref = prob_of_profit(legs, credit, snap.spot, snap.atm_iv, _dte(exp))
    assert pop is not None and flat_ref is not None
    assert pop < flat_ref  # skew (richer OTM puts) shifts mass down -> lower PoP


def test_estimate_prob_profit_falls_back_to_flat_when_no_iv() -> None:
    exp = _expiry(30)
    legs, snap = _legs(exp), _snapshot(_chain(exp, with_iv=False))
    strat = BullPutSpread()
    pop = strat.estimate_prob_profit(legs, snap)
    credit = strat.estimate_credit(legs, snap)
    flat_ref = prob_of_profit(legs, credit, snap.spot, snap.atm_iv, _dte(exp))
    assert pop is not None and flat_ref is not None
    assert abs(pop - flat_ref) < 1e-9  # no smile -> identical to the flat path


def test_estimate_expected_value_uses_smile_and_lowers_ev() -> None:
    exp = _expiry(30)
    legs, snap = _legs(exp), _snapshot(_chain(exp, with_iv=True))
    strat = BullPutSpread()
    ev = strat.estimate_expected_value(legs, snap)
    credit = strat.estimate_credit(legs, snap)
    flat_ref = expected_value_dollars(legs, credit, snap.spot, snap.hv20, _dte(exp))
    assert ev is not None and flat_ref is not None
    assert ev < flat_ref  # skewed downside tail lowers EV vs the flat realized-vol model
