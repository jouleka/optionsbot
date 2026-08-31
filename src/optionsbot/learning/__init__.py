"""Versioned learning components for managed intraday option outcomes.

Nothing in this package owns broker authority.  It turns immutable shadow
observations into auditable challenger artifacts; promotion and execution
remain explicit, deterministic decisions.
"""

from optionsbot.learning.managed_model import (
    ContextIncrementalReport,
    ManagedModelArtifact,
    ManagedPrediction,
    ManagedSample,
    PromotionPolicy,
    PromotionReport,
    ProspectiveReport,
    compare_context_incremental_value,
    evaluate_prospective_rows,
    fit_managed_model,
    predict_managed_outcome,
    score_frozen_artifact,
    walk_forward_evaluate,
)

__all__ = [
    "ManagedModelArtifact",
    "ManagedPrediction",
    "ManagedSample",
    "ProspectiveReport",
    "PromotionPolicy",
    "PromotionReport",
    "ContextIncrementalReport",
    "compare_context_incremental_value",
    "fit_managed_model",
    "evaluate_prospective_rows",
    "predict_managed_outcome",
    "score_frozen_artifact",
    "walk_forward_evaluate",
]
