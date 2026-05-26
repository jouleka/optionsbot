"""Iron Butterfly: defined-risk neutral short-premium with both shorts at ATM.

Legs (one expiry, ~45 DTE by default):
  - Short put at ATM (strike closest to spot)
  - Short call at ATM (same strike as the short put)
  - Long put one wing below the body (~5% of spot)
  - Long call one wing above the body (~5% of spot)

Profile: neutral direction, high IV. Tighter / higher-reward / higher-risk than
the Iron Condor because both shorts collapse onto a single ATM strike.
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
)

_DEFAULT_WING_WIDTH_PCT = 0.05  # 5% of spot, rounded to nearest available strike


class IronButterfly(Strategy):
    name: ClassVar[str] = "iron_butterfly"
    display_name: ClassVar[str] = "Iron Butterfly"
    defined_risk: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "high")}
    )
    factor_weights: ClassVar[dict[str, float]] = {
        "iv_rank": 0.30,
        "iv_hv": 0.20,
        "liquidity": 0.20,
        "dte_match": 0.10,
        "earnings_penalty": 0.10,
        "range_bound": 0.10,
    }

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
        if expiry is None:
            return None
        chain_at_expiry = filter_by_expiry(snapshot.chain, expiry)
        puts = filter_by_right(chain_at_expiry, "P")
        calls = filter_by_right(chain_at_expiry, "C")
        # ATM body: snap to the closest available strike on each side (same value
        # if the chain is symmetric, which it is for index/ETF chains in $5
        # increments around a round-number spot).
        atm_put = closest_strike(puts, snapshot.spot)
        atm_call = closest_strike(calls, snapshot.spot)
        if atm_put is None or atm_call is None:
            return None
        # If puts and calls landed on different strikes (asymmetric chain), pick
        # the same strike for both shorts so the butterfly body is a single point.
        body_strike = atm_call.strike  # callers can override by passing a denser chain
        short_put = next((leg for leg in puts if leg.strike == body_strike), None)
        if short_put is None:
            # Fallback: use whatever the put side landed on.
            short_put = atm_put
            body_strike = short_put.strike
            short_call = next(
                (leg for leg in calls if leg.strike == body_strike), atm_call
            )
        else:
            short_call = atm_call
        # Wings ~5% of spot from the body, snapping to the nearest available strike.
        wing_width = max(1.0, snapshot.spot * _DEFAULT_WING_WIDTH_PCT)
        long_put_target = body_strike - wing_width
        long_call_target = body_strike + wing_width
        long_put = min(
            (leg for leg in puts if leg.strike < body_strike),
            key=lambda leg: abs(leg.strike - long_put_target),
            default=None,
        )
        long_call = min(
            (leg for leg in calls if leg.strike > body_strike),
            key=lambda leg: abs(leg.strike - long_call_target),
            default=None,
        )
        if long_put is None or long_call is None:
            return None
        return (
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=long_put.strike,
                right="P",
            ),
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=expiry,
                strike=short_put.strike,
                right="P",
            ),
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=expiry,
                strike=short_call.strike,
                right="C",
            ),
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=expiry,
                strike=long_call.strike,
                right="C",
            ),
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
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

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        long_put = next(
            (leg for leg in legs if leg.side == "buy" and leg.right == "P"), None
        )
        short_put = next(
            (leg for leg in legs if leg.side == "sell" and leg.right == "P"), None
        )
        short_call = next(
            (leg for leg in legs if leg.side == "sell" and leg.right == "C"), None
        )
        long_call = next(
            (leg for leg in legs if leg.side == "buy" and leg.right == "C"), None
        )
        if not (long_put and short_put and short_call and long_call):
            return None
        if any(
            leg.strike is None for leg in (long_put, short_put, short_call, long_call)
        ):
            return None
        assert long_put.strike is not None
        assert short_put.strike is not None
        assert short_call.strike is not None
        assert long_call.strike is not None
        put_width = short_put.strike - long_put.strike
        call_width = long_call.strike - short_call.strike
        max_width = max(put_width, call_width)
        credit = self.estimate_credit(legs, snapshot)
        return max_width * 100 - credit

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Spot pinned exactly at the body strike -> keep the full credit.
        return self.estimate_credit(legs, snapshot)
