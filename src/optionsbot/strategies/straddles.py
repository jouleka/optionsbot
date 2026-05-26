"""Straddles and strangles: two-leg same-expiry volatility plays.

Four strategies, all 2 legs of opposite ``right`` sharing one expiry:

* :class:`LongStraddle` -- buy ATM call + buy ATM put. Defined risk (debit
  paid). Wants neutral direction and LOW IV (cheap entry). ``long_premium``.
* :class:`LongStrangle` -- buy OTM call (~+0.30 delta) + buy OTM put
  (~-0.30 delta). Same financial shape as the straddle but cheaper because
  both legs are further OTM.
* :class:`ShortStraddle` -- sell ATM call + sell ATM put. UNDEFINED RISK
  (max loss is None). Wants neutral direction and HIGH IV (premium-rich).
  Rationale appends a recommendation to consider the Iron Butterfly as the
  defined-risk alternative.
* :class:`ShortStrangle` -- sell OTM call (~+0.16 delta) + sell OTM put
  (~-0.16 delta). Same shape as Short Straddle. Rationale recommends the
  Iron Condor as the defined-risk alternative.

All four share the same credit-from-mids formula and use ``closest_strike``
or ``find_strike_by_delta`` (depending on whether ATM or by-delta).
"""

from __future__ import annotations

from typing import ClassVar

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.strategies.base import Leg, Strategy, StrategySnapshot
from optionsbot.strategies.strikes import (
    closest_expiry_to_dte,
    closest_strike,
    filter_by_expiry,
    filter_by_right,
    find_strike_by_delta,
)

# Strangle-leg delta targets (magnitude; sign applied per right).
_LONG_STRANGLE_DELTA = 0.30
_SHORT_STRANGLE_DELTA = 0.16


def _estimate_credit(
    legs: tuple[Leg, ...], snapshot: StrategySnapshot
) -> float:
    """Net credit (sells contribute +) per single contract set, in dollars."""
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


_LONG_PREMIUM_WEIGHTS: dict[str, float] = {
    "iv_rank": 0.30,
    "iv_hv": 0.20,
    "liquidity": 0.15,
    "dte_match": 0.15,
    "earnings_penalty": 0.0,
    "range_bound": 0.20,
}


_SHORT_PREMIUM_WEIGHTS: dict[str, float] = {
    "iv_rank": 0.40,
    "iv_hv": 0.25,
    "liquidity": 0.15,
    "dte_match": 0.10,
    "earnings_penalty": 0.05,
    "range_bound": 0.05,
}


# ---------------------------------------------------------------------------
# Long Straddle (IBK-31)
# ---------------------------------------------------------------------------


class LongStraddle(Strategy):
    name: ClassVar[str] = "long_straddle"
    display_name: ClassVar[str] = "Long Straddle"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "low")}
    )
    factor_weights: ClassVar[dict[str, float]] = _LONG_PREMIUM_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        calls = filter_by_right(chain_at_expiry, "C")
        puts = filter_by_right(chain_at_expiry, "P")
        atm_call = closest_strike(calls, snapshot.spot)
        atm_put = closest_strike(puts, snapshot.spot)
        if atm_call is None or atm_put is None:
            return None
        # Snap both legs to the call-side ATM strike for a clean single-strike
        # straddle; fall back to whatever the put side picked if asymmetric.
        body = atm_call.strike
        matching_put = next(
            (leg for leg in puts if leg.strike == body), None
        )
        if matching_put is None:
            matching_put = atm_put
            body = matching_put.strike
            matching_call = next(
                (leg for leg in calls if leg.strike == body), atm_call
            )
        else:
            matching_call = atm_call
        return (
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=matching_call.strike,
                right="C",
            ),
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=matching_put.strike,
                right="P",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Long-premium: max loss = absolute debit. Credit is negative for debits.
        credit = self.estimate_credit(legs, snapshot)
        return -credit if credit < 0 else 0.0


# ---------------------------------------------------------------------------
# Long Strangle (IBK-32)
# ---------------------------------------------------------------------------


class LongStrangle(Strategy):
    name: ClassVar[str] = "long_strangle"
    display_name: ClassVar[str] = "Long Strangle"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "low")}
    )
    factor_weights: ClassVar[dict[str, float]] = _LONG_PREMIUM_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        calls = filter_by_right(chain_at_expiry, "C")
        puts = filter_by_right(chain_at_expiry, "P")
        call_leg = find_strike_by_delta(calls, _LONG_STRANGLE_DELTA, "C")
        put_leg = find_strike_by_delta(puts, -_LONG_STRANGLE_DELTA, "P")
        if call_leg is None or put_leg is None:
            return None
        return (
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=call_leg.strike,
                right="C",
            ),
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
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        credit = self.estimate_credit(legs, snapshot)
        return -credit if credit < 0 else 0.0


# ---------------------------------------------------------------------------
# Short Straddle (IBK-33) -- UNDEFINED RISK
# ---------------------------------------------------------------------------


class ShortStraddle(Strategy):
    name: ClassVar[str] = "short_straddle"
    display_name: ClassVar[str] = "Short Straddle"
    defined_risk: ClassVar[bool] = False
    long_premium: ClassVar[bool] = False
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "high")}
    )
    factor_weights: ClassVar[dict[str, float]] = _SHORT_PREMIUM_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        calls = filter_by_right(chain_at_expiry, "C")
        puts = filter_by_right(chain_at_expiry, "P")
        atm_call = closest_strike(calls, snapshot.spot)
        atm_put = closest_strike(puts, snapshot.spot)
        if atm_call is None or atm_put is None:
            return None
        body = atm_call.strike
        matching_put = next(
            (leg for leg in puts if leg.strike == body), None
        )
        if matching_put is None:
            matching_put = atm_put
            body = matching_put.strike
            matching_call = next(
                (leg for leg in calls if leg.strike == body), atm_call
            )
        else:
            matching_call = atm_call
        return (
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=expiry,
                strike=matching_call.strike,
                right="C",
            ),
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=expiry,
                strike=matching_put.strike,
                right="P",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        return None  # naked short premium -- unlimited on both sides

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        return self.estimate_credit(legs, snapshot)

    def _build_rationale(
        self,
        snapshot: StrategySnapshot,
        legs: tuple[Leg, ...],
        credit: float,
        max_loss: float | None,
    ) -> str:
        base = super()._build_rationale(snapshot, legs, credit, max_loss)
        return (
            f"{base} UNDEFINED RISK -- consider Iron Butterfly as "
            "defined-risk alternative."
        )


# ---------------------------------------------------------------------------
# Short Strangle (IBK-34) -- UNDEFINED RISK
# ---------------------------------------------------------------------------


class ShortStrangle(Strategy):
    name: ClassVar[str] = "short_strangle"
    display_name: ClassVar[str] = "Short Strangle"
    defined_risk: ClassVar[bool] = False
    long_premium: ClassVar[bool] = False
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "high")}
    )
    factor_weights: ClassVar[dict[str, float]] = _SHORT_PREMIUM_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        calls = filter_by_right(chain_at_expiry, "C")
        puts = filter_by_right(chain_at_expiry, "P")
        call_leg = find_strike_by_delta(calls, _SHORT_STRANGLE_DELTA, "C")
        put_leg = find_strike_by_delta(puts, -_SHORT_STRANGLE_DELTA, "P")
        if call_leg is None or put_leg is None:
            return None
        return (
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=expiry,
                strike=call_leg.strike,
                right="C",
            ),
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=expiry,
                strike=put_leg.strike,
                right="P",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        return None

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        return self.estimate_credit(legs, snapshot)

    def _build_rationale(
        self,
        snapshot: StrategySnapshot,
        legs: tuple[Leg, ...],
        credit: float,
        max_loss: float | None,
    ) -> str:
        base = super()._build_rationale(snapshot, legs, credit, max_loss)
        return (
            f"{base} UNDEFINED RISK -- consider Iron Condor as "
            "defined-risk alternative."
        )
