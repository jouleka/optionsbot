"""Dataclasses for the validation backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from optionsbot.strategies.base import Leg


@dataclass(frozen=True, slots=True)
class PickRecord:
    """One recorded pick, reconstructed from strategy_scores + snapshots."""

    symbol: str
    entry_spot: float
    entry_date: date
    expiry: str
    dte_days: int
    legs: tuple[Leg, ...]
    credit_or_debit: float
    prob_profit: float
    score: float
    strategy: str


@dataclass(frozen=True, slots=True)
class BacktestRow:
    """Per-pick backtest outcome: model prediction vs realized win-rates."""

    symbol: str
    strategy: str
    predicted: float
    raw: float
    dedrift: float
    n: int


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_raw: float
    mean_dedrift: float


@dataclass(frozen=True, slots=True)
class BacktestReport:
    buckets: tuple[CalibrationBucket, ...]
    by_strategy: dict[str, CalibrationBucket] = field(default_factory=dict)
    overall_count: int = 0
    overall_mean_pred: float = 0.0
    overall_mean_raw: float = 0.0
    overall_mean_dedrift: float = 0.0


@dataclass(frozen=True, slots=True)
class UnevaluatedPick:
    strategy_score_id: int
    symbol: str
    strategy: str
    expiry: str
    entry_spot: float
    legs: tuple[Leg, ...]
    credit_or_debit: float
    predicted_prob_profit: float | None
    score: float
    max_profit: float | None
    max_loss: float | None
    risk_tier: str


@dataclass(frozen=True, slots=True)
class OutcomeGroup:
    label: str
    count: int
    win_rate: float
    mean_pred_pop: float
    total_pnl: float
    avg_pnl: float


@dataclass(frozen=True, slots=True)
class OutcomesReport:
    overall: OutcomeGroup
    by_strategy: dict[str, OutcomeGroup]
    by_risk_tier: dict[str, OutcomeGroup]
