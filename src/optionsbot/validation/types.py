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
