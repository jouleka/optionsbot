"""JSON-serialization helpers for MCP tool responses."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from optionsbot.analysis.types import MarketView
from optionsbot.scoring import ScoredStrategy
from optionsbot.strategies import Leg


def dump_view(view: MarketView) -> dict[str, Any]:
    return {
        "direction": view.direction,
        "direction_strength": view.direction_strength,
        "iv_regime": view.iv_regime,
        "iv_rank_value": view.iv_rank_value,
        "earnings_in_window": view.earnings_in_window,
        "warming_up": view.warming_up,
    }


def dump_leg(leg: Leg) -> dict[str, Any]:
    return asdict(leg)


def dump_scored(s: ScoredStrategy) -> dict[str, Any]:
    return {
        "strategy_name": s.strategy_name,
        "score": s.score,
        "rationale": s.rationale,
        "factors": s.factors.as_dict(),
        "suggestion": {
            "legs": [dump_leg(leg) for leg in s.suggestion.legs],
            "credit_or_debit": s.suggestion.credit_or_debit,
            "max_loss": s.suggestion.max_loss,
            "max_profit": s.suggestion.max_profit,
            "prob_profit": s.suggestion.prob_profit,
            "suggested_quantity": s.suggestion.suggested_quantity,
            "defined_risk": s.suggestion.defined_risk,
        },
    }
