"""Tests for the four vertical spreads (Bull Put, Bear Call, Bull Call, Bear Put).

Four tests per vertical: applicable views, leg shape (2 legs of correct right with
short/long ordered correctly), max-loss formula, factor weights sum to 1.0.
"""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.verticals import (
    BearCallSpread,
    BearPutSpread,
    BullCallSpread,
    BullPutSpread,
)
from tests.unit.test_strategies.conftest import make_view


def _snapshot(
    chain: tuple[OptionChainLeg, ...], view: MarketView
) -> StrategySnapshot:
    return StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain,
        view=view,
    )


# ---------------------------------------------------------------------------
# Bull Put Spread (credit)
# ---------------------------------------------------------------------------


def test_bull_put_applicable_to_bull_high_and_bull_neutral() -> None:
    bp = BullPutSpread()
    assert bp.is_applicable(make_view("bull", "high"))
    assert bp.is_applicable(make_view("bull", "neutral"))
    assert not bp.is_applicable(make_view("bull", "low"))
    assert not bp.is_applicable(make_view("neutral", "high"))
    assert not bp.is_applicable(make_view("bear", "high"))


def test_bull_put_two_put_legs_short_above_long(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bp = BullPutSpread()
    legs = bp.suggest_legs(_snapshot(chain_45dte, make_view("bull", "high")))
    assert legs is not None
    assert len(legs) == 2
    assert all(leg.right == "P" for leg in legs)
    by_side = {leg.side: leg for leg in legs}
    assert set(by_side) == {"buy", "sell"}
    short = by_side["sell"]
    long_ = by_side["buy"]
    assert short.strike is not None and long_.strike is not None
    # Bull put credit: short higher-strike put, long lower-strike put (hedge)
    assert short.strike > long_.strike


def test_bull_put_max_loss_matches_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bp = BullPutSpread()
    snap = _snapshot(chain_45dte, make_view("bull", "high"))
    legs = bp.suggest_legs(snap)
    assert legs is not None
    credit = bp.estimate_credit(legs, snap)
    assert credit > 0  # credit spread
    max_loss = bp.estimate_max_loss(legs, snap)
    by_side = {leg.side: leg for leg in legs}
    short_k = by_side["sell"].strike
    long_k = by_side["buy"].strike
    assert short_k is not None and long_k is not None
    assert max_loss is not None
    assert max_loss == pytest.approx((short_k - long_k) * 100 - credit)


def test_bull_put_factor_weights_sum_to_one() -> None:
    assert sum(BullPutSpread().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bear Call Spread (credit)
# ---------------------------------------------------------------------------


def test_bear_call_applicable_to_bear_high_and_bear_neutral() -> None:
    bc = BearCallSpread()
    assert bc.is_applicable(make_view("bear", "high"))
    assert bc.is_applicable(make_view("bear", "neutral"))
    assert not bc.is_applicable(make_view("bear", "low"))
    assert not bc.is_applicable(make_view("bull", "high"))


def test_bear_call_two_call_legs_short_below_long(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bc = BearCallSpread()
    legs = bc.suggest_legs(_snapshot(chain_45dte, make_view("bear", "high")))
    assert legs is not None
    assert len(legs) == 2
    assert all(leg.right == "C" for leg in legs)
    by_side = {leg.side: leg for leg in legs}
    assert set(by_side) == {"buy", "sell"}
    short = by_side["sell"]
    long_ = by_side["buy"]
    assert short.strike is not None and long_.strike is not None
    # Bear call credit: short lower-strike call, long higher-strike call (hedge)
    assert short.strike < long_.strike


def test_bear_call_max_loss_matches_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bc = BearCallSpread()
    snap = _snapshot(chain_45dte, make_view("bear", "high"))
    legs = bc.suggest_legs(snap)
    assert legs is not None
    credit = bc.estimate_credit(legs, snap)
    assert credit > 0  # credit spread
    max_loss = bc.estimate_max_loss(legs, snap)
    by_side = {leg.side: leg for leg in legs}
    short_k = by_side["sell"].strike
    long_k = by_side["buy"].strike
    assert short_k is not None and long_k is not None
    assert max_loss is not None
    assert max_loss == pytest.approx((long_k - short_k) * 100 - credit)


def test_bear_call_factor_weights_sum_to_one() -> None:
    assert sum(BearCallSpread().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bull Call Spread (debit)
# ---------------------------------------------------------------------------


def test_bull_call_applicable_to_bull_low_and_bull_neutral() -> None:
    bc = BullCallSpread()
    assert bc.is_applicable(make_view("bull", "low"))
    assert bc.is_applicable(make_view("bull", "neutral"))
    assert not bc.is_applicable(make_view("bull", "high"))
    assert not bc.is_applicable(make_view("bear", "low"))


def test_bull_call_two_call_legs_long_below_short(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bc = BullCallSpread()
    legs = bc.suggest_legs(_snapshot(chain_45dte, make_view("bull", "low")))
    assert legs is not None
    assert len(legs) == 2
    assert all(leg.right == "C" for leg in legs)
    by_side = {leg.side: leg for leg in legs}
    assert set(by_side) == {"buy", "sell"}
    long_ = by_side["buy"]
    short = by_side["sell"]
    assert long_.strike is not None and short.strike is not None
    # Bull call debit: long lower-strike (ATM-ish), short higher-strike OTM
    assert long_.strike < short.strike


def test_bull_call_debit_max_loss_is_absolute_debit(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bc = BullCallSpread()
    snap = _snapshot(chain_45dte, make_view("bull", "low"))
    legs = bc.suggest_legs(snap)
    assert legs is not None
    credit = bc.estimate_credit(legs, snap)
    assert credit < 0  # debit spread: we pay
    max_loss = bc.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss > 0  # positive (absolute value of debit)
    assert max_loss == pytest.approx(-credit)
    # max_profit = (short - long) * 100 - debit
    by_side = {leg.side: leg for leg in legs}
    short_k = by_side["sell"].strike
    long_k = by_side["buy"].strike
    assert short_k is not None and long_k is not None
    max_profit = bc.estimate_max_profit(legs, snap)
    assert max_profit is not None
    assert max_profit == pytest.approx((short_k - long_k) * 100 - (-credit))


def test_bull_call_factor_weights_sum_to_one() -> None:
    assert sum(BullCallSpread().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bear Put Spread (debit)
# ---------------------------------------------------------------------------


def test_bear_put_applicable_to_bear_low_and_bear_neutral() -> None:
    bp = BearPutSpread()
    assert bp.is_applicable(make_view("bear", "low"))
    assert bp.is_applicable(make_view("bear", "neutral"))
    assert not bp.is_applicable(make_view("bear", "high"))
    assert not bp.is_applicable(make_view("bull", "low"))


def test_bear_put_two_put_legs_long_above_short(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bp = BearPutSpread()
    legs = bp.suggest_legs(_snapshot(chain_45dte, make_view("bear", "low")))
    assert legs is not None
    assert len(legs) == 2
    assert all(leg.right == "P" for leg in legs)
    by_side = {leg.side: leg for leg in legs}
    assert set(by_side) == {"buy", "sell"}
    long_ = by_side["buy"]
    short = by_side["sell"]
    assert long_.strike is not None and short.strike is not None
    # Bear put debit: long higher-strike (ATM-ish), short lower-strike OTM
    assert long_.strike > short.strike


def test_bear_put_debit_max_loss_is_absolute_debit(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    bp = BearPutSpread()
    snap = _snapshot(chain_45dte, make_view("bear", "low"))
    legs = bp.suggest_legs(snap)
    assert legs is not None
    credit = bp.estimate_credit(legs, snap)
    assert credit < 0  # debit
    max_loss = bp.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss > 0
    assert max_loss == pytest.approx(-credit)


def test_bear_put_factor_weights_sum_to_one() -> None:
    assert sum(BearPutSpread().factor_weights.values()) == pytest.approx(1.0)
