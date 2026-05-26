"""Covered Call (IBK-41) and Cash-Secured Put (IBK-42).

Both are short-premium strategies that lean on existing capital -- either
an established 100-share equity position (Covered Call) or sufficient
cash to absorb assignment (Cash-Secured Put).

* :class:`CoveredCall` -- long 100 shares + short OTM call at ~+0.30 delta.
  Eligibility is gated on ``snapshot.position.position >= 100``; the
  IBKR positions layer (IBK-77) populates ``snapshot.position``. The
  returned leg tuple includes a ``sec_type="STK"`` leg representing the
  *existing* holding so the formatter can render the full structure -- the
  shares are not a new purchase, and they're excluded from
  :meth:`CoveredCall.estimate_credit`.

* :class:`CashSecuredPut` -- short OTM put at ~-0.30 delta. Overrides
  :meth:`Strategy.suggest_size` to additionally require
  ``account_value >= max_loss_per_contract`` (defensible interpretation
  of "cash to cover" assignment for at least one contract).
"""

from __future__ import annotations

from typing import ClassVar

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.strategies.base import Leg, Strategy, StrategySnapshot
from optionsbot.strategies.strikes import (
    closest_expiry_to_dte,
    filter_by_expiry,
    filter_by_right,
    find_strike_by_delta,
)

_SHORT_CALL_DELTA = 0.30  # covered call short delta target
_SHORT_PUT_DELTA = -0.30  # cash-secured put short delta target (puts: negative)


def _estimate_option_credit(
    legs: tuple[Leg, ...], snapshot: StrategySnapshot
) -> float:
    """Net option-only credit (sells contribute +) per single contract set.

    Stock legs (``sec_type="STK"`` and/or option-key fields ``None``) are
    skipped -- for Covered Call the shares already exist, so they don't
    contribute to the trade's net credit/debit.
    """
    chain_by_key = {
        (leg.expiry, leg.strike, leg.right): leg for leg in snapshot.chain
    }
    total = 0.0
    for leg in legs:
        if leg.expiry is None or leg.strike is None or leg.right is None:
            continue
        chain_leg = chain_by_key.get((leg.expiry, leg.strike, leg.right))
        if chain_leg is None or chain_leg.bid is None or chain_leg.ask is None:
            continue
        mid = (chain_leg.bid + chain_leg.ask) / 2
        total += mid if leg.side == "sell" else -mid
    return total * 100


# ---------------------------------------------------------------------------
# Covered Call (IBK-41)
# ---------------------------------------------------------------------------


class CoveredCall(Strategy):
    name: ClassVar[str] = "covered_call"
    display_name: ClassVar[str] = "Covered Call"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = False
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {
            ("bull", "high"),
            ("neutral", "high"),
            ("bull", "neutral"),
        }
    )
    factor_weights: ClassVar[dict[str, float]] = {
        "iv_rank": 0.30,
        "iv_hv": 0.15,
        "liquidity": 0.20,
        "dte_match": 0.10,
        "earnings_penalty": 0.20,
        "range_bound": 0.05,
    }

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        # Eligibility: need 100+ shares to write a single contract.
        if snapshot.position is None or snapshot.position.position < 100:
            return None
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        calls = filter_by_right(
            filter_by_expiry(snapshot.chain, expiry), "C"
        )
        short_call = find_strike_by_delta(calls, _SHORT_CALL_DELTA, "C")
        if short_call is None:
            return None
        # Stock leg represents the *existing* long position (NOT a new buy).
        # estimate_credit skips it so we don't double-count the share cost.
        stock_leg = Leg(
            symbol=snapshot.symbol,
            side="buy",
            sec_type="STK",
            expiry=None,
            strike=None,
            right=None,
            quantity=100,
        )
        call_leg = Leg(
            symbol=snapshot.symbol,
            side="sell",
            sec_type="OPT",
            expiry=expiry,
            strike=short_call.strike,
            right="C",
        )
        return (stock_leg, call_leg)

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        # Only the short call contributes -- shares already owned.
        return _estimate_option_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Worst case: stock to 0. We lose position * spot, less the call premium.
        if snapshot.position is None:
            return None
        credit = self.estimate_credit(legs, snapshot)
        return snapshot.position.position * snapshot.spot - credit

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Capped at assignment: stock called away at call_strike.
        # Profit = position * (call_strike - avg_cost) + credit.
        if snapshot.position is None:
            return None
        call_leg = next(
            (leg for leg in legs if leg.sec_type == "OPT" and leg.side == "sell"),
            None,
        )
        if call_leg is None or call_leg.strike is None:
            return None
        credit = self.estimate_credit(legs, snapshot)
        return (
            snapshot.position.position * (call_leg.strike - snapshot.position.avg_cost)
            + credit
        )


# ---------------------------------------------------------------------------
# Cash-Secured Put (IBK-42)
# ---------------------------------------------------------------------------


class CashSecuredPut(Strategy):
    name: ClassVar[str] = "cash_secured_put"
    display_name: ClassVar[str] = "Cash-Secured Put"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = False
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bull", "high"), ("neutral", "high")}
    )
    factor_weights: ClassVar[dict[str, float]] = {
        "iv_rank": 0.30,
        "iv_hv": 0.20,
        "liquidity": 0.20,
        "dte_match": 0.10,
        "earnings_penalty": 0.15,
        "range_bound": 0.05,
    }

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        puts = filter_by_right(
            filter_by_expiry(snapshot.chain, expiry), "P"
        )
        short_put = find_strike_by_delta(puts, _SHORT_PUT_DELTA, "P")
        if short_put is None:
            return None
        return (
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                sec_type="OPT",
                expiry=expiry,
                strike=short_put.strike,
                right="P",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_option_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Worst case: assigned then stock to 0. Loss = put_strike * 100 - credit.
        short_put = next(
            (leg for leg in legs if leg.side == "sell" and leg.right == "P"), None
        )
        if short_put is None or short_put.strike is None:
            return None
        credit = self.estimate_credit(legs, snapshot)
        return short_put.strike * 100 - credit

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Put expires worthless -> we keep the credit.
        return self.estimate_credit(legs, snapshot)

    def suggest_size(
        self,
        account_value: float,
        max_loss_per_contract: float | None,
        risk_pct: float = 0.02,
    ) -> int:
        """Standard risk-budget sizing PLUS a cash-to-cover floor.

        On top of the base sizing, require ``account_value >=
        max_loss_per_contract`` so we have enough cash to absorb at least
        one assignment. This is a defensible interpretation of "cash to
        cover" -- a stricter rule (cash >= put_strike * 100 * size for
        every unit) is possible but rejected here for simplicity; the
        risk-budget cap below should keep total exposure proportional
        to the account anyway.
        """
        base_size = super().suggest_size(account_value, max_loss_per_contract, risk_pct)
        if base_size == 0:
            return 0
        if max_loss_per_contract is None:
            return 0
        if account_value < max_loss_per_contract:
            return 0
        return base_size
