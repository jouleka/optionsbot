"""Unit tests for the composite scoring engine + top-K selector."""

from __future__ import annotations

import pytest

from optionsbot.scoring.composite import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    compute_composite,
    compute_factor_breakdown,
    score_all,
    score_strategy,
    top_k,
)
from optionsbot.scoring.types import FactorBreakdown, ScoredStrategy
from optionsbot.strategies import (
    IronCondor,
    LongCall,
    StrategySnapshot,
    StrategySuggestion,
    all_strategies,
)
from optionsbot.strategies.base import Leg

# ---------------------------------------------------------------------------
# compute_factor_breakdown
# ---------------------------------------------------------------------------


def test_compute_factor_breakdown_returns_all_six_fields_in_unit_range(
    base_snapshot: StrategySnapshot,
    base_suggestion: StrategySuggestion,
    base_strategy: IronCondor,
) -> None:
    breakdown = compute_factor_breakdown(
        base_snapshot, base_suggestion, base_strategy
    )
    assert isinstance(breakdown, FactorBreakdown)
    for value in breakdown.as_dict().values():
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# compute_composite
# ---------------------------------------------------------------------------


def test_compute_composite_perfect_factors_with_unit_weight_sum_is_100() -> None:
    breakdown = FactorBreakdown(
        iv_rank=1.0,
        iv_hv=1.0,
        liquidity=1.0,
        dte_match=1.0,
        earnings_penalty=1.0,
        range_bound=1.0,
    )
    weights = {
        "iv_rank": 0.2,
        "iv_hv": 0.2,
        "liquidity": 0.2,
        "dte_match": 0.2,
        "earnings_penalty": 0.1,
        "range_bound": 0.1,
    }
    assert compute_composite(breakdown, weights) == pytest.approx(100.0)


def test_compute_composite_zero_factors_is_zero() -> None:
    breakdown = FactorBreakdown(
        iv_rank=0.0,
        iv_hv=0.0,
        liquidity=0.0,
        dte_match=0.0,
        earnings_penalty=0.0,
        range_bound=0.0,
    )
    weights = {
        "iv_rank": 0.2,
        "iv_hv": 0.2,
        "liquidity": 0.2,
        "dte_match": 0.2,
        "earnings_penalty": 0.1,
        "range_bound": 0.1,
    }
    assert compute_composite(breakdown, weights) == 0.0


def test_compute_composite_clips_above_100_when_weights_misbehave() -> None:
    # All-1.0 factors with weights summing to 2.0 would naively give 200; expect clip to 100.
    breakdown = FactorBreakdown(
        iv_rank=1.0,
        iv_hv=1.0,
        liquidity=1.0,
        dte_match=1.0,
        earnings_penalty=1.0,
        range_bound=1.0,
    )
    weights = {
        "iv_rank": 0.4,
        "iv_hv": 0.4,
        "liquidity": 0.4,
        "dte_match": 0.4,
        "earnings_penalty": 0.2,
        "range_bound": 0.2,
    }
    assert compute_composite(breakdown, weights) == 100.0


# ---------------------------------------------------------------------------
# score_strategy
# ---------------------------------------------------------------------------


def test_score_strategy_returns_none_for_inapplicable_view(
    base_snapshot: StrategySnapshot,
) -> None:
    # base_snapshot is neutral/high-IV; LongCall requires bull/{low,neutral}.
    scored = score_strategy(base_snapshot, LongCall())
    assert scored is None


def test_score_strategy_returns_scored_strategy_for_applicable_view(
    base_snapshot: StrategySnapshot,
) -> None:
    scored = score_strategy(base_snapshot, IronCondor(), account_value=100_000)
    assert scored is not None
    assert isinstance(scored, ScoredStrategy)
    assert scored.strategy_name == "iron_condor"
    assert 0.0 <= scored.score <= 100.0
    assert scored.rationale != ""  # filled in by build_rationale
    assert isinstance(scored.factors, FactorBreakdown)


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------


def test_score_all_skips_strategies_that_cannot_produce_suggestions(
    base_snapshot: StrategySnapshot,
) -> None:
    results = score_all(base_snapshot, account_value=100_000)
    # Every result corresponds to a strategy that produced a suggestion.
    assert len(results) >= 1
    assert all(isinstance(r, ScoredStrategy) for r in results)
    # No LongCall (bull-only) in a neutral snapshot.
    names = {r.strategy_name for r in results}
    assert "long_call" not in names
    assert "long_put" not in names


def test_score_all_accepts_custom_strategy_pool(
    base_snapshot: StrategySnapshot,
) -> None:
    pool: tuple = (IronCondor(),)
    results = score_all(base_snapshot, account_value=100_000, strategies=pool)
    assert len(results) == 1
    assert results[0].strategy_name == "iron_condor"


def test_score_all_default_pool_is_all_strategies(
    base_snapshot: StrategySnapshot,
) -> None:
    # Sanity: at most len(all_strategies()) results.
    results = score_all(base_snapshot, account_value=100_000)
    assert len(results) <= len(all_strategies())


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


def _make_scored(name: str, score: float) -> ScoredStrategy:
    """Build a ScoredStrategy with a minimal placeholder suggestion + factors."""
    return ScoredStrategy(
        strategy_name=name,
        score=score,
        factors=FactorBreakdown(
            iv_rank=0.0,
            iv_hv=0.0,
            liquidity=0.0,
            dte_match=0.0,
            earnings_penalty=0.0,
            range_bound=0.0,
        ),
        suggestion=StrategySuggestion(
            strategy_name=name,
            legs=(Leg(symbol="SPY", side="buy", sec_type="STK", quantity=100),),
            credit_or_debit=0.0,
            max_loss=None,
            max_profit=None,
            prob_profit=None,
            suggested_quantity=0,
            defined_risk=True,
            rationale="",
        ),
        rationale="",
    )


def test_top_k_filters_by_threshold() -> None:
    scored = (
        _make_scored("a", 80.0),
        _make_scored("b", 65.0),
        _make_scored("c", 90.0),
    )
    result = top_k(scored, k=3, threshold=70.0)
    names = [r.strategy_name for r in result]
    assert "b" not in names
    assert set(names) == {"a", "c"}


def test_top_k_sorts_descending_by_score() -> None:
    scored = (
        _make_scored("a", 75.0),
        _make_scored("b", 92.0),
        _make_scored("c", 85.0),
    )
    result = top_k(scored, k=3, threshold=70.0)
    assert [r.score for r in result] == [92.0, 85.0, 75.0]


def test_top_k_returns_at_most_k_entries() -> None:
    scored = (
        _make_scored("a", 95.0),
        _make_scored("b", 90.0),
        _make_scored("c", 85.0),
        _make_scored("d", 80.0),
        _make_scored("e", 75.0),
    )
    result = top_k(scored, k=2, threshold=70.0)
    assert len(result) == 2
    assert [r.strategy_name for r in result] == ["a", "b"]


def test_top_k_returns_empty_when_nothing_crosses_threshold() -> None:
    scored = (
        _make_scored("a", 50.0),
        _make_scored("b", 30.0),
    )
    result = top_k(scored, k=3, threshold=70.0)
    assert result == ()


def test_top_k_defaults_are_module_constants() -> None:
    assert DEFAULT_TOP_K == 3
    assert DEFAULT_THRESHOLD == 70.0
