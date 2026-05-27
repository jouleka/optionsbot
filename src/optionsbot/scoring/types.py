"""Output dataclasses for the scoring engine."""

from __future__ import annotations

from dataclasses import dataclass

from optionsbot.strategies import Strategy, StrategySnapshot, StrategySuggestion


@dataclass(frozen=True, slots=True)
class FactorBreakdown:
    """Per-factor score breakdown. All values in ``[0.0, 1.0]``."""

    iv_rank: float
    iv_hv: float
    liquidity: float
    dte_match: float
    earnings_penalty: float
    range_bound: float

    def as_dict(self) -> dict[str, float]:
        return {
            "iv_rank": self.iv_rank,
            "iv_hv": self.iv_hv,
            "liquidity": self.liquidity,
            "dte_match": self.dte_match,
            "earnings_penalty": self.earnings_penalty,
            "range_bound": self.range_bound,
        }


@dataclass(frozen=True, slots=True)
class FactorContext:
    """Bundle passed to every factor function so signatures stay uniform."""

    snapshot: StrategySnapshot
    suggestion: StrategySuggestion
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class ScoredStrategy:
    """A strategy's composite score plus its breakdown and the suggestion.

    ``rationale`` is filled in by
    :func:`optionsbot.scoring.rationale.build_rationale`.
    """

    strategy_name: str
    score: float  # 0..100
    factors: FactorBreakdown
    suggestion: StrategySuggestion
    rationale: str
