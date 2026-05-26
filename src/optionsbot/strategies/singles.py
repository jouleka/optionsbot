"""Long Call (IBK-43) and Long Put (IBK-44): single-leg directional debits.

Both are the simplest of the 16 strategies -- one option leg, bought, at
roughly ATM (~+/-0.50 delta), targeting ~45 DTE. They are ``long_premium``
trades: max loss equals the debit paid; max profit is unbounded.

* :class:`LongCall` -- buy 1 call at ~+0.50 delta. Applicable to bullish
  views in low/neutral IV (cheap entry for upside exposure).
* :class:`LongPut` -- buy 1 put at ~-0.50 delta. Mirror of Long Call for
  bearish views in low/neutral IV.

Both share a common ``_estimate_debit_credit`` helper that prices the
single leg off the chain's bid/ask midpoint. Returns a negative number
(per :class:`StrategySuggestion` convention: positive = credit,
negative = debit).
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

# ATM-ish target delta for the single bought option. Spec also allows
# ~+/-0.40 for a slightly OTM cheaper entry; we use 0.50 as the default
# since it's both the textbook ATM straight-debit setup and what the chain
# fixture's symmetric ramp lands cleanly on.
_LONG_SINGLE_DELTA = 0.50


def _estimate_debit_credit(
    legs: tuple[Leg, ...], snapshot: StrategySnapshot
) -> float:
    """Net credit (sells contribute +) per single contract set, in dollars.

    For a long single, the only leg is a buy, so the return is always
    non-positive (a debit). Shared shape with the multi-leg helpers in
    :mod:`optionsbot.strategies.straddles`.
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


# Long-premium weights for single-leg debits. iv_rank gets less weight here
# than for straddles/strangles since the directional bet is the primary
# driver; range_bound is also lower because directional debits don't depend
# on the underlying staying range-bound. liquidity matters more for singles
# because a single bad fill swings the whole trade.
_LONG_SINGLE_WEIGHTS: dict[str, float] = {
    "iv_rank": 0.20,
    "iv_hv": 0.20,
    "liquidity": 0.20,
    "dte_match": 0.20,
    "earnings_penalty": 0.10,
    "range_bound": 0.10,
}


# ---------------------------------------------------------------------------
# Long Call (IBK-43)
# ---------------------------------------------------------------------------


class LongCall(Strategy):
    name: ClassVar[str] = "long_call"
    display_name: ClassVar[str] = "Long Call"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bull", "low"), ("bull", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _LONG_SINGLE_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        calls = filter_by_right(chain_at_expiry, "C")
        call_leg = find_strike_by_delta(calls, _LONG_SINGLE_DELTA, "C")
        if call_leg is None:
            return None
        return (
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=call_leg.strike,
                right="C",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_debit_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Long-premium: max loss = absolute debit. Credit is negative for debits.
        credit = self.estimate_credit(legs, snapshot)
        return -credit if credit < 0 else 0.0


# ---------------------------------------------------------------------------
# Long Put (IBK-44)
# ---------------------------------------------------------------------------


class LongPut(Strategy):
    name: ClassVar[str] = "long_put"
    display_name: ClassVar[str] = "Long Put"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bear", "low"), ("bear", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _LONG_SINGLE_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        puts = filter_by_right(chain_at_expiry, "P")
        # Puts have negative delta; pass the signed target.
        put_leg = find_strike_by_delta(puts, -_LONG_SINGLE_DELTA, "P")
        if put_leg is None:
            return None
        return (
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=put_leg.strike,
                right="P",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_debit_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        credit = self.estimate_credit(legs, snapshot)
        return -credit if credit < 0 else 0.0
