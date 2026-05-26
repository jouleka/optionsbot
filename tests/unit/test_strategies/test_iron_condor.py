"""Tests for Iron Condor strategy."""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.iron_condor import IronCondor


def test_iron_condor_applicable_to_neutral_high_iv(
    neutral_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    assert ic.is_applicable(neutral_high_iv_view)


def test_iron_condor_not_applicable_to_bull_high_iv(
    bullish_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    assert not ic.is_applicable(bullish_high_iv_view)


def test_iron_condor_returns_four_legs(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ic.suggest_legs(snap)
    assert legs is not None
    assert len(legs) == 4
    sides_rights = sorted((leg.side, leg.right) for leg in legs)
    assert sides_rights == sorted(
        [("buy", "P"), ("sell", "P"), ("sell", "C"), ("buy", "C")]
    )


def test_iron_condor_strikes_ordered(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ic.suggest_legs(snap)
    assert legs is not None
    by_role = {(leg.side, leg.right): leg for leg in legs}
    assert by_role[("buy", "P")].strike is not None
    assert by_role[("sell", "P")].strike is not None
    assert by_role[("sell", "C")].strike is not None
    assert by_role[("buy", "C")].strike is not None
    assert by_role[("buy", "P")].strike < by_role[("sell", "P")].strike
    assert by_role[("sell", "P")].strike < by_role[("sell", "C")].strike
    assert by_role[("sell", "C")].strike < by_role[("buy", "C")].strike


def test_iron_condor_max_loss_matches_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ic.suggest_legs(snap)
    assert legs is not None
    max_loss = ic.estimate_max_loss(legs, snap)
    credit = ic.estimate_credit(legs, snap)
    # max loss should be positive and less than max wing width * 100
    assert max_loss is not None
    assert max_loss > 0
    by_role = {(leg.side, leg.right): leg for leg in legs}
    sp = by_role[("sell", "P")].strike
    lp = by_role[("buy", "P")].strike
    lc = by_role[("buy", "C")].strike
    sc = by_role[("sell", "C")].strike
    assert sp is not None and lp is not None and lc is not None and sc is not None
    put_width = sp - lp
    call_width = lc - sc
    assert max_loss == pytest.approx(max(put_width, call_width) * 100 - credit)


def test_iron_condor_credit_positive_for_short_premium(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    legs = ic.suggest_legs(snap)
    assert legs is not None
    credit = ic.estimate_credit(legs, snap)
    assert credit > 0


def test_iron_condor_factor_weights_sum_to_one() -> None:
    ic = IronCondor()
    assert sum(ic.factor_weights.values()) == pytest.approx(1.0)


def test_iron_condor_build_suggestion_returns_full_record(
    chain_45dte: tuple[OptionChainLeg, ...],
    neutral_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=neutral_high_iv_view,
    )
    suggestion = ic.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert suggestion.strategy_name == "iron_condor"
    assert len(suggestion.legs) == 4
    assert suggestion.defined_risk is True
    assert suggestion.suggested_quantity >= 1
    assert "Iron Condor" in suggestion.rationale


def test_iron_condor_build_suggestion_none_for_inapplicable_view(
    chain_45dte: tuple[OptionChainLeg, ...],
    bullish_high_iv_view: MarketView,
) -> None:
    ic = IronCondor()
    snap = StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain_45dte,
        view=bullish_high_iv_view,
    )
    assert ic.build_suggestion(snap, account_value=100_000) is None
