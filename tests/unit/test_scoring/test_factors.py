"""Unit tests for the six per-strategy factor calculators."""

from __future__ import annotations

from dataclasses import replace

import pytest

from optionsbot.scoring.factors import (
    dte_match_score,
    earnings_penalty,
    iv_hv_score,
    iv_rank_score,
    liquidity_score,
    range_bound_score,
)
from optionsbot.scoring.types import FactorContext
from optionsbot.strategies import (
    LongStraddle,
    StrategySnapshot,
    StrategySuggestion,
)
from optionsbot.strategies.base import Leg, Strategy
from optionsbot.strategies.iron_condor import IronCondor
from tests.unit.test_scoring.conftest import make_view


def _ctx(
    snap: StrategySnapshot,
    sugg: StrategySuggestion,
    strat: Strategy,
) -> FactorContext:
    return FactorContext(snapshot=snap, suggestion=sugg, strategy=strat)


# ---------------------------------------------------------------------------
# iv_rank_score
# ---------------------------------------------------------------------------


def test_iv_rank_high_is_good_for_short_premium(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    assert iv_rank_score(_ctx(base_snapshot, base_suggestion, base_strategy)) == pytest.approx(0.75)


def test_iv_rank_high_is_bad_for_long_premium(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
) -> None:
    long_straddle = LongStraddle()
    score = iv_rank_score(_ctx(base_snapshot, base_suggestion, long_straddle))
    assert score == pytest.approx(0.25)  # 1.0 - 0.75


def test_iv_rank_none_returns_neutral(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, iv_rank=None)
    assert iv_rank_score(_ctx(snap, base_suggestion, base_strategy)) == 0.5


# ---------------------------------------------------------------------------
# iv_hv_score
# ---------------------------------------------------------------------------


def test_iv_hv_ratio_125_maps_above_half_for_short_premium(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # IV/HV = 0.25/0.20 = 1.25 -> (1.25-0.5)/1.0 = 0.75 (in [0,1])
    score = iv_hv_score(_ctx(base_snapshot, base_suggestion, base_strategy))
    assert score == pytest.approx(0.75)


def test_iv_hv_ratio_inverts_for_long_premium(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
) -> None:
    long_straddle = LongStraddle()
    # short-premium short would be 0.75; inverted -> 0.25
    score = iv_hv_score(_ctx(base_snapshot, base_suggestion, long_straddle))
    assert score == pytest.approx(0.25)


def test_iv_hv_missing_hv_returns_neutral(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, hv20=None)
    assert iv_hv_score(_ctx(snap, base_suggestion, base_strategy)) == 0.5


def test_iv_hv_zero_hv_returns_neutral(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, hv20=0.0)
    assert iv_hv_score(_ctx(snap, base_suggestion, base_strategy)) == 0.5


def test_iv_hv_clips_at_one(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # Ratio = 2.0 -> raw = 1.5 -> clipped to 1.0
    snap = replace(base_snapshot, atm_iv=0.40, hv20=0.20)
    assert iv_hv_score(_ctx(snap, base_suggestion, base_strategy)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# liquidity_score
# ---------------------------------------------------------------------------


def test_liquidity_score_in_range(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    score = liquidity_score(_ctx(base_snapshot, base_suggestion, base_strategy))
    assert 0.0 <= score <= 1.0
    # bid=2.0/ask=2.2 -> spread=0.20, mid=2.1 -> spread_pct ~= 0.0952 -> spread_score ~= 0.048
    # OI=1000 -> oi_score=1.0. Per-leg = (0.048 + 1.0)/2 ~= 0.524 for all 4 option legs.
    assert score == pytest.approx((1.0 - 0.20 / 2.1 / 0.10 + 1.0) / 2.0, rel=1e-3)


def test_liquidity_returns_zero_when_no_option_legs(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # Build a stock-only suggestion (no OPT legs in the suggestion's legs).
    stock_only = replace(
        base_suggestion,
        legs=(
            Leg(
                symbol="SPY",
                side="buy",
                sec_type="STK",
                quantity=100,
            ),
        ),
    )
    score = liquidity_score(_ctx(base_snapshot, stock_only, base_strategy))
    assert score == 0.0


# ---------------------------------------------------------------------------
# dte_match_score
# ---------------------------------------------------------------------------


def test_dte_match_perfect_score(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # base_snapshot dte_target=45 and the suggestion's legs are at 45 DTE expiry
    score = dte_match_score(_ctx(base_snapshot, base_suggestion, base_strategy))
    assert score == pytest.approx(1.0)


def test_dte_match_decays_with_distance(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # Move target by 15 days -> closest = 15 -> score = 1 - 15/30 = 0.5
    snap = replace(base_snapshot, dte_target=60)
    score = dte_match_score(_ctx(snap, base_suggestion, base_strategy))
    assert score == pytest.approx(0.5)


def test_dte_match_no_option_legs_returns_neutral(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # No legs with an expiry -> 0.5 neutral
    stock_only = replace(
        base_suggestion,
        legs=(
            Leg(
                symbol="SPY",
                side="buy",
                sec_type="STK",
                quantity=100,
            ),
        ),
    )
    score = dte_match_score(_ctx(base_snapshot, stock_only, base_strategy))
    assert score == 0.5


def test_dte_match_floor_at_zero(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    # Move target so far that closest delta exceeds 30 days -> clip to 0.0
    snap = replace(base_snapshot, dte_target=200)
    score = dte_match_score(_ctx(snap, base_suggestion, base_strategy))
    assert score == 0.0


# ---------------------------------------------------------------------------
# earnings_penalty
# ---------------------------------------------------------------------------


def test_earnings_penalty_no_earnings_is_one(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    assert earnings_penalty(_ctx(base_snapshot, base_suggestion, base_strategy)) == 1.0


def test_earnings_penalty_short_premium_with_earnings_is_zero(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, view=make_view(earnings_in_window=True))
    assert earnings_penalty(_ctx(snap, base_suggestion, base_strategy)) == 0.0


def test_earnings_penalty_long_premium_with_earnings_is_one(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
) -> None:
    long_straddle = LongStraddle()
    snap = replace(base_snapshot, view=make_view(earnings_in_window=True))
    assert earnings_penalty(_ctx(snap, base_suggestion, long_straddle)) == 1.0


# ---------------------------------------------------------------------------
# range_bound_score
# ---------------------------------------------------------------------------


def test_range_bound_neutral_weak_is_one(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, view=make_view(direction="neutral", strength="weak"))
    assert range_bound_score(_ctx(snap, base_suggestion, base_strategy)) == 1.0


def test_range_bound_neutral_strong_is_half(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, view=make_view(direction="neutral", strength="strong"))
    assert range_bound_score(_ctx(snap, base_suggestion, base_strategy)) == 0.5


def test_range_bound_directional_weak_is_low(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, view=make_view(direction="bull", strength="weak"))
    assert range_bound_score(_ctx(snap, base_suggestion, base_strategy)) == pytest.approx(0.3)


def test_range_bound_directional_strong_is_zero(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    snap = replace(base_snapshot, view=make_view(direction="bear", strength="strong"))
    assert range_bound_score(_ctx(snap, base_suggestion, base_strategy)) == 0.0
