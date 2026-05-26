"""Tests for Calendar Spread (IBK-39) and Diagonal Spread (IBK-40).

Both are 2-leg multi-expiry constructions sharing a debit-pay financial shape
(max loss = debit) and the same ``factor_weights``. They differ in strike
selection:

* Calendar -- both legs at the SAME ATM strike, different expiries.
* Diagonal -- DIFFERENT strikes, different expiries, with the right
  (call vs put) chosen by ``snapshot.view.direction``.

Coverage:

1. ``test_calendar_applicable_to_neutral_low_iv`` -- ``is_applicable`` matches
   the ``frozenset({("neutral","low"), ("neutral","neutral")})`` membership.
2. ``test_calendar_returns_two_legs_with_different_expiries_same_strike``.
3. ``test_calendar_legs_are_calls`` -- default-right is call.
4. ``test_calendar_max_loss_matches_debit`` -- ``max_loss == |credit|`` when
   ``credit < 0``.
5. ``test_diagonal_applicable_to_bull_low_iv``.
6. ``test_diagonal_legs_use_calls_when_bullish_view`` and the neutral case.
7. ``test_diagonal_legs_use_puts_when_bearish_view``.
8. ``test_diagonal_legs_have_different_strikes_and_expiries``.
9. ``test_calendar_factor_weights_sum_to_one``.
10. ``test_diagonal_factor_weights_sum_to_one``.
"""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.calendar import CalendarSpread, DiagonalSpread
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
# Calendar Spread (IBK-39)
# ---------------------------------------------------------------------------


def test_calendar_applicable_to_neutral_low_iv() -> None:
    s = CalendarSpread()
    assert s.is_applicable(make_view("neutral", "low"))
    assert s.is_applicable(make_view("neutral", "neutral"))
    assert not s.is_applicable(make_view("neutral", "high"))
    assert not s.is_applicable(make_view("bull", "low"))
    assert not s.is_applicable(make_view("bear", "low"))
    assert s.long_premium is False


def test_calendar_returns_two_legs_with_different_expiries_same_strike(
    chain_multi_dte: tuple[OptionChainLeg, ...],
) -> None:
    s = CalendarSpread()
    legs = s.suggest_legs(_snapshot(chain_multi_dte, make_view("neutral", "low")))
    assert legs is not None
    assert len(legs) == 2
    # Different expiries
    expiries = {leg.expiry for leg in legs}
    assert len(expiries) == 2
    # Same strike (ATM = 400 in the fixture)
    strikes = {leg.strike for leg in legs}
    assert strikes == {400.0}
    # One buy (back, longer-dated) and one sell (front, near-term)
    sides = sorted(leg.side for leg in legs)
    assert sides == ["buy", "sell"]


def test_calendar_legs_are_calls(
    chain_multi_dte: tuple[OptionChainLeg, ...],
) -> None:
    s = CalendarSpread()
    legs = s.suggest_legs(_snapshot(chain_multi_dte, make_view("neutral", "low")))
    assert legs is not None
    rights = {leg.right for leg in legs}
    assert rights == {"C"}


def test_calendar_max_loss_matches_debit(
    chain_multi_dte: tuple[OptionChainLeg, ...],
) -> None:
    s = CalendarSpread()
    snap = _snapshot(chain_multi_dte, make_view("neutral", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    # Back leg is longer DTE -> richer -> we pay more for it than we collect on
    # the short front. Net is a debit.
    assert credit < 0
    max_loss = s.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss == pytest.approx(-credit)
    # max_profit is intentionally undefined for the calendar.
    assert s.estimate_max_profit(legs, snap) is None


def test_calendar_factor_weights_sum_to_one() -> None:
    assert sum(CalendarSpread().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Diagonal Spread (IBK-40)
# ---------------------------------------------------------------------------


def test_diagonal_applicable_to_bull_low_iv() -> None:
    s = DiagonalSpread()
    assert s.is_applicable(make_view("bull", "low"))
    assert s.is_applicable(make_view("bear", "low"))
    assert s.is_applicable(make_view("neutral", "low"))
    assert not s.is_applicable(make_view("bull", "high"))
    assert not s.is_applicable(make_view("neutral", "high"))
    assert s.long_premium is False


def test_diagonal_legs_use_calls_when_bullish_view(
    chain_multi_dte: tuple[OptionChainLeg, ...],
) -> None:
    s = DiagonalSpread()
    legs = s.suggest_legs(_snapshot(chain_multi_dte, make_view("bull", "low")))
    assert legs is not None
    assert len(legs) == 2
    rights = {leg.right for leg in legs}
    assert rights == {"C"}
    # Neutral also resolves to calls.
    legs_neutral = s.suggest_legs(
        _snapshot(chain_multi_dte, make_view("neutral", "low"))
    )
    assert legs_neutral is not None
    assert {leg.right for leg in legs_neutral} == {"C"}


def test_diagonal_legs_use_puts_when_bearish_view(
    chain_multi_dte: tuple[OptionChainLeg, ...],
) -> None:
    s = DiagonalSpread()
    legs = s.suggest_legs(_snapshot(chain_multi_dte, make_view("bear", "low")))
    assert legs is not None
    assert len(legs) == 2
    rights = {leg.right for leg in legs}
    assert rights == {"P"}


def test_diagonal_legs_have_different_strikes_and_expiries(
    chain_multi_dte: tuple[OptionChainLeg, ...],
) -> None:
    s = DiagonalSpread()
    legs = s.suggest_legs(_snapshot(chain_multi_dte, make_view("bull", "low")))
    assert legs is not None
    strikes = {leg.strike for leg in legs}
    expiries = {leg.expiry for leg in legs}
    assert len(strikes) == 2
    assert len(expiries) == 2
    # The diagonal is a debit spread (back-leg longer DTE costs more than front).
    snap = _snapshot(chain_multi_dte, make_view("bull", "low"))
    credit = s.estimate_credit(legs, snap)
    assert credit < 0
    max_loss = s.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss == pytest.approx(-credit)
    assert s.estimate_max_profit(legs, snap) is None


def test_diagonal_factor_weights_sum_to_one() -> None:
    assert sum(DiagonalSpread().factor_weights.values()) == pytest.approx(1.0)
