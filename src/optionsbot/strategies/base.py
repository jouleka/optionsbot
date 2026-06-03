"""Strategy ABC plus the supporting frozen dataclasses (Leg, StrategySnapshot,
StrategySuggestion).

Every concrete strategy subclasses :class:`Strategy`, declares its applicable
market views and factor weights via :class:`~typing.ClassVar` attributes, and
implements :meth:`Strategy.suggest_legs`, :meth:`Strategy.estimate_credit`, and
:meth:`Strategy.estimate_max_loss`. The default
:meth:`Strategy.build_suggestion` orchestrates the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import ClassVar, Literal

from optionsbot.analysis.types import Direction, IVRegime, MarketView
from optionsbot.ibkr.types import OptionChainLeg, OptionRight, PositionRecord

# Risk-tier thresholds (probability of profit). Module constants for now; if these
# ever need to be user-tunable, thread them from settings the way risk_pct is.
_RISK_TIER_CONSERVATIVE_PROB = 0.65
_RISK_TIER_AGGRESSIVE_PROB = 0.40


def _expected_value(
    prob_profit: float | None,
    max_profit: float | None,
    max_loss: float | None,
    *,
    defined_risk: bool,
) -> float | None:
    """Heuristic expectancy in dollars: prob*max_profit - (1-prob)*max_loss.

    Only defined for defined-risk strategies with a bounded max_profit and a
    known prob_profit + max_loss. Returns None otherwise (NOT a claim of edge).
    """
    if not defined_risk or prob_profit is None or max_profit is None or max_loss is None:
        return None
    return prob_profit * max_profit - (1.0 - prob_profit) * max_loss


def _risk_tier(*, defined_risk: bool, prob_profit: float | None) -> str:
    """conservative / balanced / aggressive from defined-risk + prob_profit."""
    if not defined_risk or (prob_profit is not None and prob_profit < _RISK_TIER_AGGRESSIVE_PROB):
        return "aggressive"
    if prob_profit is not None and prob_profit >= _RISK_TIER_CONSERVATIVE_PROB:
        return "conservative"
    return "balanced"


@dataclass(frozen=True, slots=True)
class Leg:
    """One leg of a multi-leg strategy.

    For stock legs (Covered Call's long 100 shares), ``expiry``/``strike``/
    ``right`` are all ``None`` and ``sec_type="STK"``.
    """

    symbol: str
    side: Literal["buy", "sell"]
    sec_type: Literal["STK", "OPT"] = "OPT"
    expiry: str | None = None
    strike: float | None = None
    right: OptionRight | None = None
    quantity: int = 1  # per "unit"; suggest_size multiplies this


@dataclass(frozen=True)
class StrategySnapshot:
    """Everything a strategy needs to decide whether/how to be applied."""

    symbol: str
    spot: float
    atm_iv: float | None
    hv20: float | None
    iv_rank: float | None
    chain: tuple[OptionChainLeg, ...]
    view: MarketView
    dte_target: int = 45
    position: PositionRecord | None = None  # used by Covered Call eligibility


@dataclass(frozen=True)
class StrategySuggestion:
    """A single, sized suggestion from a strategy ready for downstream scoring."""

    strategy_name: str
    legs: tuple[Leg, ...]
    credit_or_debit: float  # positive credit, negative debit, per single contract set
    max_loss: float | None  # absolute value per single contract set; None = undefined risk
    max_profit: float | None  # None = unbounded
    prob_profit: float | None  # None when can't estimate
    suggested_quantity: int  # contracts (or share-sets for stock-leg strategies)
    defined_risk: bool
    rationale: str
    reward_risk: float | None = None  # max_profit / max_loss; None if either is None
    expected_value: float | None = None  # heuristic expectancy in dollars; None if not computable
    risk_tier: str = "balanced"  # conservative | balanced | aggressive


class Strategy(ABC):
    """Stateless strategy.

    Subclasses set class-level metadata + implement
    ``suggest_legs`` / ``estimate_credit`` / ``estimate_max_loss``.
    """

    name: ClassVar[str]  # snake_case, registry key
    display_name: ClassVar[str]  # human-readable
    defined_risk: ClassVar[bool]
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]]
    # True for strategies that net-pay premium (long straddle/strangle,
    # long call/put, long calendar/diagonal). The IBK-5 scoring engine
    # inverts the iv_rank factor for these (long-premium wants LOW iv_rank,
    # short-premium wants HIGH).
    long_premium: ClassVar[bool] = False
    # Default factor weights consumed by IBK-5 scoring engine.
    # Standard factor names: iv_rank, iv_hv, liquidity, dte_match,
    # earnings_penalty, range_bound. Must sum to 1.0.
    factor_weights: ClassVar[dict[str, float]]

    def is_applicable(self, view: MarketView) -> bool:
        return (view.direction, view.iv_regime) in self.applicable_views

    @abstractmethod
    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        """Return concrete legs, or None if no usable strikes/expiries in the chain."""

    @abstractmethod
    def estimate_credit(self, legs: tuple[Leg, ...], snapshot: StrategySnapshot) -> float:
        """Net credit (positive) or debit (negative) per single contract set."""

    @abstractmethod
    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        """Absolute max loss per single contract set. None means undefined."""

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        """Default: None (unbounded). Override for defined-profit strategies (spreads)."""
        return None

    def estimate_prob_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        """P(profit) at expiry via the lognormal payoff model.

        Returns None for non-modelable positions (stock legs, or legs spanning
        more than one expiry -- calendars/diagonals). Computes the front DTE
        from the (single) leg expiry.
        """
        from optionsbot.scoring.payoff import is_terminal_modelable, prob_of_profit

        if not is_terminal_modelable(legs):
            return None
        expiry = legs[0].expiry
        assert expiry is not None  # guaranteed by is_terminal_modelable
        dte_days = (datetime.strptime(expiry, "%Y%m%d").date() - date.today()).days
        credit = self.estimate_credit(legs, snapshot)
        return prob_of_profit(legs, credit, snapshot.spot, snapshot.atm_iv, float(dte_days))

    def suggest_size(
        self,
        account_value: float,
        max_loss_per_contract: float | None,
        risk_pct: float = 0.02,
    ) -> int:
        """Default: budget * risk_pct / max_loss_per_contract, floored at 1.

        Returns 0 if max_loss is None (undefined risk) or non-positive.
        """
        if max_loss_per_contract is None or max_loss_per_contract <= 0:
            return 0
        budget = account_value * risk_pct
        return max(1, int(budget // max_loss_per_contract))

    def build_suggestion(
        self,
        snapshot: StrategySnapshot,
        account_value: float | None = None,
        risk_pct: float = 0.02,
    ) -> StrategySuggestion | None:
        """High-level: call suggest_legs, fill in metrics, build a Suggestion.

        Returns None when the strategy can't produce legs (chain doesn't have
        strikes at the required deltas, view isn't applicable, etc.).
        """
        if not self.is_applicable(snapshot.view):
            return None
        legs = self.suggest_legs(snapshot)
        if legs is None:
            return None
        credit = self.estimate_credit(legs, snapshot)
        max_loss = self.estimate_max_loss(legs, snapshot)
        max_profit = self.estimate_max_profit(legs, snapshot)
        reward_risk = (
            max_profit / max_loss
            if max_profit is not None and max_loss is not None and max_loss > 0
            else None
        )
        prob_profit = self.estimate_prob_profit(legs, snapshot)
        expected_value = _expected_value(
            prob_profit, max_profit, max_loss, defined_risk=self.defined_risk
        )
        risk_tier = _risk_tier(defined_risk=self.defined_risk, prob_profit=prob_profit)
        size = (
            self.suggest_size(account_value, max_loss, risk_pct)
            if account_value is not None
            else 0
        )
        return StrategySuggestion(
            strategy_name=self.name,
            legs=legs,
            credit_or_debit=credit,
            max_loss=max_loss,
            max_profit=max_profit,
            prob_profit=prob_profit,
            suggested_quantity=size,
            defined_risk=self.defined_risk,
            rationale=self._build_rationale(snapshot, legs, credit, max_loss),
            reward_risk=reward_risk,
            expected_value=expected_value,
            risk_tier=risk_tier,
        )

    def _build_rationale(
        self,
        snapshot: StrategySnapshot,
        legs: tuple[Leg, ...],
        credit: float,
        max_loss: float | None,
    ) -> str:
        """Default rationale text. Strategies can override for richer prose."""
        ml = f"${max_loss:.2f}" if max_loss is not None else "undefined"
        kind = "credit" if credit > 0 else "debit"
        return (
            f"{self.display_name} on {snapshot.symbol} at "
            f"view={snapshot.view.direction}/{snapshot.view.iv_regime}, "
            f"{len(legs)} legs, net {kind} ${abs(credit):.2f}, max loss {ml}."
        )
