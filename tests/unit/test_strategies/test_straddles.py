"""Tests for Long/Short Straddle and Long/Short Strangle strategies.

Five tests per strategy (20 total):

1. ``applicable_to_expected_view`` -- ``is_applicable`` returns True for the
   ``(direction, iv_regime)`` tuple in ``applicable_views`` and False for a
   sample of neighbors.
2. ``returns_two_legs_with_correct_sides_and_rights`` -- 1 call + 1 put,
   both on the same ``side`` (buy for longs, sell for shorts).
3. ``max_loss_matches_formula_or_is_none`` -- the two long variants have a
   positive ``max_loss`` equal to the absolute debit paid; the two short
   variants return ``None`` (undefined risk).
4. ``defined_risk_flag_matches_class`` -- ``s.defined_risk`` reads through
   the suggestion record.
5. ``rationale_warning_for_undefined_risk`` -- the two shorts append
   "UNDEFINED RISK" to the rationale and name the defined-risk alternative
   (Iron Butterfly / Iron Condor); the two longs do NOT mention it.

Extra coverage for the two long variants: assert call/put strike location
(ATM for straddle, ~+/-0.30 delta OTM for strangle).
"""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.straddles import (
    LongStraddle,
    LongStrangle,
    ShortStraddle,
    ShortStrangle,
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
        iv_rank=view.iv_rank_value or 0.5,
        chain=chain,
        view=view,
    )


# ---------------------------------------------------------------------------
# Long Straddle (IBK-31)
# ---------------------------------------------------------------------------


def test_long_straddle_applicable_to_neutral_low_iv() -> None:
    s = LongStraddle()
    assert s.is_applicable(make_view("neutral", "low"))
    assert not s.is_applicable(make_view("neutral", "high"))
    assert not s.is_applicable(make_view("neutral", "neutral"))
    assert not s.is_applicable(make_view("bull", "low"))
    assert s.long_premium is True


def test_long_straddle_returns_two_legs_one_call_one_put_both_buy(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStraddle()
    legs = s.suggest_legs(_snapshot(chain_45dte, make_view("neutral", "low")))
    assert legs is not None
    assert len(legs) == 2
    sides = {leg.side for leg in legs}
    rights = sorted(leg.right for leg in legs if leg.right is not None)
    assert sides == {"buy"}
    assert rights == ["C", "P"]
    # Both legs share the ATM strike (closest to spot=400 -> 400)
    strikes = {leg.strike for leg in legs}
    assert strikes == {400.0}
    # And the same expiry
    expiries = {leg.expiry for leg in legs}
    assert len(expiries) == 1


def test_long_straddle_max_loss_equals_absolute_debit(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStraddle()
    snap = _snapshot(chain_45dte, make_view("neutral", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    assert credit < 0  # debit
    max_loss = s.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss > 0
    assert max_loss == pytest.approx(-credit)
    # Unbounded profit
    assert s.estimate_max_profit(legs, snap) is None


def test_long_straddle_defined_risk_flag_true(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStraddle()
    snap = _snapshot(chain_45dte, make_view("neutral", "low"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert suggestion.defined_risk is True
    assert s.defined_risk is True


def test_long_straddle_rationale_has_no_undefined_risk_warning(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStraddle()
    snap = _snapshot(chain_45dte, make_view("neutral", "low"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert "UNDEFINED RISK" not in suggestion.rationale


def test_long_straddle_factor_weights_sum_to_one() -> None:
    assert sum(LongStraddle().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Long Strangle (IBK-32)
# ---------------------------------------------------------------------------


def test_long_strangle_applicable_to_neutral_low_iv() -> None:
    s = LongStrangle()
    assert s.is_applicable(make_view("neutral", "low"))
    assert not s.is_applicable(make_view("neutral", "high"))
    assert not s.is_applicable(make_view("bull", "low"))
    assert s.long_premium is True


def test_long_strangle_returns_two_legs_one_call_one_put_both_buy_otm(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStrangle()
    legs = s.suggest_legs(_snapshot(chain_45dte, make_view("neutral", "low")))
    assert legs is not None
    assert len(legs) == 2
    sides = {leg.side for leg in legs}
    assert sides == {"buy"}
    by_right = {leg.right: leg for leg in legs}
    assert set(by_right) == {"C", "P"}
    # OTM: call strike above spot, put strike below spot
    call_leg = by_right["C"]
    put_leg = by_right["P"]
    assert call_leg.strike is not None and put_leg.strike is not None
    assert call_leg.strike > 400.0
    assert put_leg.strike < 400.0
    # Same expiry
    assert call_leg.expiry == put_leg.expiry


def test_long_strangle_max_loss_equals_absolute_debit(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStrangle()
    snap = _snapshot(chain_45dte, make_view("neutral", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    assert credit < 0  # debit
    max_loss = s.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss > 0
    assert max_loss == pytest.approx(-credit)
    assert s.estimate_max_profit(legs, snap) is None


def test_long_strangle_defined_risk_flag_true(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStrangle()
    snap = _snapshot(chain_45dte, make_view("neutral", "low"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert suggestion.defined_risk is True
    assert s.defined_risk is True


def test_long_strangle_rationale_has_no_undefined_risk_warning(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongStrangle()
    snap = _snapshot(chain_45dte, make_view("neutral", "low"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert "UNDEFINED RISK" not in suggestion.rationale


def test_long_strangle_factor_weights_sum_to_one() -> None:
    assert sum(LongStrangle().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Short Straddle (IBK-33) -- UNDEFINED RISK
# ---------------------------------------------------------------------------


def test_short_straddle_applicable_to_neutral_high_iv() -> None:
    s = ShortStraddle()
    assert s.is_applicable(make_view("neutral", "high"))
    assert not s.is_applicable(make_view("neutral", "low"))
    assert not s.is_applicable(make_view("neutral", "neutral"))
    assert not s.is_applicable(make_view("bull", "high"))
    assert s.long_premium is False


def test_short_straddle_returns_two_legs_one_call_one_put_both_sell_atm(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStraddle()
    legs = s.suggest_legs(_snapshot(chain_45dte, make_view("neutral", "high")))
    assert legs is not None
    assert len(legs) == 2
    sides = {leg.side for leg in legs}
    rights = sorted(leg.right for leg in legs if leg.right is not None)
    assert sides == {"sell"}
    assert rights == ["C", "P"]
    # Both ATM (closest to spot 400)
    strikes = {leg.strike for leg in legs}
    assert strikes == {400.0}


def test_short_straddle_max_loss_is_none_credit_positive(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStraddle()
    snap = _snapshot(chain_45dte, make_view("neutral", "high"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    assert credit > 0  # short premium = receive
    assert s.estimate_max_loss(legs, snap) is None  # undefined
    # max_profit = credit
    assert s.estimate_max_profit(legs, snap) == pytest.approx(credit)


def test_short_straddle_defined_risk_flag_false(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStraddle()
    snap = _snapshot(chain_45dte, make_view("neutral", "high"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert suggestion.defined_risk is False
    assert s.defined_risk is False
    # Undefined risk -> suggest_size returns 0
    assert suggestion.suggested_quantity == 0


def test_short_straddle_rationale_warns_undefined_risk_recommends_iron_butterfly(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStraddle()
    snap = _snapshot(chain_45dte, make_view("neutral", "high"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert "UNDEFINED RISK" in suggestion.rationale
    assert "Iron Butterfly" in suggestion.rationale


def test_short_straddle_factor_weights_sum_to_one() -> None:
    assert sum(ShortStraddle().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Short Strangle (IBK-34) -- UNDEFINED RISK
# ---------------------------------------------------------------------------


def test_short_strangle_applicable_to_neutral_high_iv() -> None:
    s = ShortStrangle()
    assert s.is_applicable(make_view("neutral", "high"))
    assert not s.is_applicable(make_view("neutral", "low"))
    assert not s.is_applicable(make_view("bull", "high"))
    assert s.long_premium is False


def test_short_strangle_returns_two_legs_one_call_one_put_both_sell_otm(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStrangle()
    legs = s.suggest_legs(_snapshot(chain_45dte, make_view("neutral", "high")))
    assert legs is not None
    assert len(legs) == 2
    sides = {leg.side for leg in legs}
    assert sides == {"sell"}
    by_right = {leg.right: leg for leg in legs}
    assert set(by_right) == {"C", "P"}
    call_leg = by_right["C"]
    put_leg = by_right["P"]
    assert call_leg.strike is not None and put_leg.strike is not None
    # Both OTM (~0.16 delta wings)
    assert call_leg.strike > 400.0
    assert put_leg.strike < 400.0
    assert call_leg.expiry == put_leg.expiry


def test_short_strangle_max_loss_is_none_credit_positive(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStrangle()
    snap = _snapshot(chain_45dte, make_view("neutral", "high"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    assert credit > 0
    assert s.estimate_max_loss(legs, snap) is None
    assert s.estimate_max_profit(legs, snap) == pytest.approx(credit)


def test_short_strangle_defined_risk_flag_false(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStrangle()
    snap = _snapshot(chain_45dte, make_view("neutral", "high"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert suggestion.defined_risk is False
    assert s.defined_risk is False
    assert suggestion.suggested_quantity == 0


def test_short_strangle_rationale_warns_undefined_risk_recommends_iron_condor(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = ShortStrangle()
    snap = _snapshot(chain_45dte, make_view("neutral", "high"))
    suggestion = s.build_suggestion(snap, account_value=100_000)
    assert suggestion is not None
    assert "UNDEFINED RISK" in suggestion.rationale
    assert "Iron Condor" in suggestion.rationale


def test_short_strangle_factor_weights_sum_to_one() -> None:
    assert sum(ShortStrangle().factor_weights.values()) == pytest.approx(1.0)
