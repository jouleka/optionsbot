"""Composite scoring engine for option strategies.

Re-exports the six factor functions, the composite formula + top-K selector,
and the output dataclasses. The rationale generator lands in a follow-up task.
"""

from optionsbot.scoring.composite import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    compute_composite,
    compute_factor_breakdown,
    score_all,
    score_strategy,
    top_k,
)
from optionsbot.scoring.factors import (
    dte_match_score,
    earnings_penalty,
    iv_hv_score,
    iv_rank_score,
    liquidity_score,
    range_bound_score,
)
from optionsbot.scoring.types import FactorBreakdown, FactorContext, ScoredStrategy

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "FactorBreakdown",
    "FactorContext",
    "ScoredStrategy",
    "compute_composite",
    "compute_factor_breakdown",
    "dte_match_score",
    "earnings_penalty",
    "iv_hv_score",
    "iv_rank_score",
    "liquidity_score",
    "range_bound_score",
    "score_all",
    "score_strategy",
    "top_k",
]
