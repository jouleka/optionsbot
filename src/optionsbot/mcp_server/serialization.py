"""JSON-serialization helpers for MCP tool responses."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optionsbot.analysis.types import MarketView
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.strategies import Leg


def iso_utc(dt: datetime | None) -> str | None:
    """Return ISO-8601 with UTC offset, even if SQLite stripped tz info.

    DateTime(timezone=True) columns round-trip through SQLite as naive
    datetimes (tzinfo stripped on read). Bare ``.isoformat()`` on a naive
    dt then loses the ``+00:00`` suffix that downstream JSON consumers
    rely on to parse it as UTC. Use this helper everywhere a SQL-returned
    timestamp is rendered to a response.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


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
            "reward_risk": s.suggestion.reward_risk,
            "expected_value": s.suggestion.expected_value,
            "risk_tier": s.suggestion.risk_tier,
        },
    }
