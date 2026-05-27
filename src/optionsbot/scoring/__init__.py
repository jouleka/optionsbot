"""Composite scoring engine for option strategies.

Re-exports the six factor functions plus the output dataclasses. The composite
formula + top-K selector and the rationale generator land in later tasks.
"""

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
    "FactorBreakdown",
    "FactorContext",
    "ScoredStrategy",
    "dte_match_score",
    "earnings_penalty",
    "iv_hv_score",
    "iv_rank_score",
    "liquidity_score",
    "range_bound_score",
]
