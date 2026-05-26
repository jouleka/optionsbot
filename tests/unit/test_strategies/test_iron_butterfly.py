"""Tests for Iron Butterfly strategy."""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.iron_butterfly import IronButterfly
from tests.unit.test_strategies.conftest import make_view


def test_iron_butterfly_applicable_only_to_neutral_high_iv(
    neutral_high_iv_view: MarketView,
) -> None:
    ib = IronButterfly()
    assert ib.is_applicable(neutral_high_iv_view)
    # Not applicable to anything else
    assert not ib.is_applicable(make_view("neutral", "neutral"))
    assert not ib.is_applicable(make_view("neutral", "low"))
    assert not ib.is_applicable(make_view("bull", "high"))


def test_iron_butterfly_short_legs_at_same_atm_strike(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ib = IronButterfly()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ib.suggest_legs(snap)
    assert legs is not None
    assert len(legs) == 4
    by_role = {(leg.side, leg.right): leg for leg in legs}
    # Short put and short call should be at the same (ATM) strike
    short_put = by_role[("sell", "P")]
    short_call = by_role[("sell", "C")]
    assert short_put.strike == short_call.strike
    # ATM strike = closest to spot 400 -> strike 400
    assert short_put.strike == 400.0


def test_iron_butterfly_strikes_ordered_with_wings_outside_body(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ib = IronButterfly()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ib.suggest_legs(snap)
    assert legs is not None
    by_role = {(leg.side, leg.right): leg for leg in legs}
    long_put = by_role[("buy", "P")]
    short_put = by_role[("sell", "P")]
    short_call = by_role[("sell", "C")]
    long_call = by_role[("buy", "C")]
    assert long_put.strike is not None
    assert short_put.strike is not None
    assert short_call.strike is not None
    assert long_call.strike is not None
    # body shorts at ATM, wings outside
    assert long_put.strike < short_put.strike
    assert short_call.strike < long_call.strike
    assert short_put.strike == short_call.strike


def test_iron_butterfly_max_loss_matches_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ib = IronButterfly()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ib.suggest_legs(snap)
    assert legs is not None
    credit = ib.estimate_credit(legs, snap)
    max_loss = ib.estimate_max_loss(legs, snap)
    by_role = {(leg.side, leg.right): leg for leg in legs}
    sp = by_role[("sell", "P")].strike
    lp = by_role[("buy", "P")].strike
    sc = by_role[("sell", "C")].strike
    lc = by_role[("buy", "C")].strike
    assert sp is not None and lp is not None and sc is not None and lc is not None
    put_width = sp - lp
    call_width = lc - sc
    assert max_loss is not None
    assert max_loss == pytest.approx(max(put_width, call_width) * 100 - credit)


def test_iron_butterfly_factor_weights_sum_to_one() -> None:
    ib = IronButterfly()
    assert sum(ib.factor_weights.values()) == pytest.approx(1.0)
