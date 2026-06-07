"""Composite scoring engine + top-K selector.

Combines the six per-strategy factors (see :mod:`optionsbot.scoring.factors`)
into a single ``0..100`` composite score, applies a strategy's class-level
``factor_weights``, and exposes a top-K selector for downstream consumers
(the daemon's alert dispatcher in IBK-7, the MCP ``analyze`` tool in IBK-6).

Functions are sync and pure: scoring is a deterministic transform of the
input :class:`~optionsbot.strategies.StrategySnapshot`.
"""

from __future__ import annotations

from typing import Literal

from optionsbot.scoring.factors import (
    dte_match_score,
    earnings_penalty,
    iv_hv_score,
    iv_rank_score,
    liquidity_score,
    range_bound_score,
)
from optionsbot.scoring.rationale import build_rationale
from optionsbot.scoring.types import FactorBreakdown, FactorContext, ScoredStrategy
from optionsbot.strategies import (
    Strategy,
    StrategySnapshot,
    StrategySuggestion,
    all_strategies,
)

# Factor calculator dispatch table -- order matches FactorBreakdown field order.
_FACTORS = (
    ("iv_rank", iv_rank_score),
    ("iv_hv", iv_hv_score),
    ("liquidity", liquidity_score),
    ("dte_match", dte_match_score),
    ("earnings_penalty", earnings_penalty),
    ("range_bound", range_bound_score),
)


DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 70.0


def edge_sort_key(suggestion: StrategySuggestion) -> tuple[int, float]:
    """Sign-aware edge ranking key. Use with ``sorted(..., reverse=True)``.

    Returns ``(tier, within_tier)``; the tier dominates the comparison so the
    within-tier value is only ever compared against the same metric:

    - tier 2 (best): positive edge (``EV > 0``) -> ordered by ``EV/max_loss`` desc
      (IBK-104 capital efficiency, valid only where EV is positive).
    - tier 1: break-even / negative (``EV <= 0``) -> ordered by RAW ``EV`` desc
      (least dollars lost first). Avoids the ``EV/max_loss`` sign-inversion, where
      dividing a small negative EV by a huge ``max_loss`` flatters it toward zero
      and floats a capital-hungry loser to the top.
    - tier 0 (last): edge ``None`` -- undefined-risk naked premium or
      non-modelable EV.
    """
    edge = suggestion.risk_normalized_expectancy
    ev = suggestion.expected_value
    if edge is None or ev is None:
        return (0, float("-inf"))
    if ev > 0:
        return (2, edge)
    return (1, ev)


def has_positive_edge(suggestion: StrategySuggestion) -> bool:
    """True when the pick has a computable positive expected value.

    Mirrors the tier-2 condition in :func:`edge_sort_key`: ``EV > 0`` with a
    defined ``max_loss`` (so ``risk_normalized_expectancy`` is not None). Used by
    the surfaces to decide the "no positive edge" state (IBK-106).
    """
    ev = suggestion.expected_value
    return (
        suggestion.risk_normalized_expectancy is not None
        and ev is not None
        and ev > 0
    )


def compute_factor_breakdown(
    snapshot: StrategySnapshot,
    suggestion: StrategySuggestion,
    strategy: Strategy,
) -> FactorBreakdown:
    """Run all six factor calculators against the snapshot + suggestion."""
    ctx = FactorContext(snapshot=snapshot, suggestion=suggestion, strategy=strategy)
    values = {name: fn(ctx) for name, fn in _FACTORS}
    return FactorBreakdown(**values)


def compute_composite(
    breakdown: FactorBreakdown,
    weights: dict[str, float],
) -> float:
    """Weighted sum of factor scores, scaled to ``[0, 100]`` and clipped.

    The naive formula is ``100 * sum(weight_i * factor_i)``. Weights are
    expected to sum to ``1.0`` (each :class:`Strategy` subclass asserts this
    in IBK-4), but the result is still clipped in case a caller passes a
    misconfigured weights dict.
    """
    factors_d = breakdown.as_dict()
    total = sum(weights.get(k, 0.0) * v for k, v in factors_d.items())
    score = total * 100.0
    return max(0.0, min(100.0, score))


def score_strategy(
    snapshot: StrategySnapshot,
    strategy: Strategy,
    account_value: float | None = None,
    risk_pct: float = 0.02,
) -> ScoredStrategy | None:
    """Build a single :class:`ScoredStrategy` for ``strategy`` on ``snapshot``.

    Returns ``None`` when the strategy can't be applied to the snapshot --
    either the view is incompatible, or
    :meth:`~optionsbot.strategies.Strategy.suggest_legs` can't find usable
    strikes/expiries in the chain.

    ``rationale`` is populated by
    :func:`optionsbot.scoring.rationale.build_rationale`.
    """
    suggestion = strategy.build_suggestion(
        snapshot, account_value=account_value, risk_pct=risk_pct
    )
    if suggestion is None:
        return None
    breakdown = compute_factor_breakdown(snapshot, suggestion, strategy)
    score = compute_composite(breakdown, strategy.factor_weights)
    rationale = build_rationale(score, breakdown, strategy)
    return ScoredStrategy(
        strategy_name=strategy.name,
        score=score,
        factors=breakdown,
        suggestion=suggestion,
        rationale=rationale,
    )


def score_all(
    snapshot: StrategySnapshot,
    account_value: float | None = None,
    risk_pct: float = 0.02,
    strategies: tuple[Strategy, ...] | None = None,
) -> tuple[ScoredStrategy, ...]:
    """Score every strategy in the pool, skipping any that return ``None``.

    Defaults to :func:`optionsbot.strategies.all_strategies` -- pass a custom
    ``strategies`` tuple to score a subset (used by tests + the MCP tool when
    callers want only a few candidates).
    """
    pool = strategies if strategies is not None else all_strategies()
    results: list[ScoredStrategy] = []
    for strategy in pool:
        scored = score_strategy(
            snapshot, strategy, account_value=account_value, risk_pct=risk_pct
        )
        if scored is not None:
            results.append(scored)
    return tuple(results)


def top_k(
    scored: tuple[ScoredStrategy, ...],
    k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    rank_by: Literal["score", "expectancy"] = "score",
) -> tuple[ScoredStrategy, ...]:
    """Return up to ``k`` strategies with ``score >= threshold``, best first.

    Always filters on the quality ``score`` threshold first (the quality gate).
    ``rank_by`` then orders the survivors: by ``score`` (default) or by
    ``expectancy`` (sign-aware edge: positive-EV picks by EV/max_loss, then
    negative-EV by raw EV, then None -- see :func:`edge_sort_key`).
    """
    filtered = [s for s in scored if s.score >= threshold]
    if rank_by == "expectancy":
        filtered.sort(key=lambda s: edge_sort_key(s.suggestion), reverse=True)
    else:
        filtered.sort(key=lambda s: s.score, reverse=True)
    return tuple(filtered[:k])
