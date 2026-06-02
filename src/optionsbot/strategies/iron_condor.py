"""Iron Condor: defined-risk neutral short-premium.

Legs (one expiry, ~45 DTE by default):
  - Short put at target_short_delta (~ -0.16)
  - Long put one width below the short put
  - Short call at target_short_delta (~ +0.16)
  - Long call one width above the short call

Profile: neutral direction, high IV (premium-rich). Defined risk.
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

_DEFAULT_SHORT_DELTA = 0.16
_DEFAULT_WING_WIDTH_PCT = 0.02  # 2% of spot, rounded to nearest available strike


class IronCondor(Strategy):
    name: ClassVar[str] = "iron_condor"
    display_name: ClassVar[str] = "Iron Condor"
    defined_risk: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "high"), ("neutral", "neutral")}
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
        # Short put at -0.16 delta (puts have negative delta), short call at +0.16
        short_put = find_strike_by_delta(puts, -_DEFAULT_SHORT_DELTA, "P")
        short_call = find_strike_by_delta(calls, _DEFAULT_SHORT_DELTA, "C")
        if short_put is None or short_call is None:
            return None
        # Wings: ~2% of spot below short put / above short call, snapping to nearest strike
        wing_width = max(1.0, snapshot.spot * _DEFAULT_WING_WIDTH_PCT)
        long_put_target = short_put.strike - wing_width
        long_call_target = short_call.strike + wing_width
        long_put = min(
            (leg for leg in puts if leg.strike < short_put.strike),
            key=lambda leg: abs(leg.strike - long_put_target),
            default=None,
        )
        long_call = min(
            (leg for leg in calls if leg.strike > short_call.strike),
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

    def estimate_credit(self, legs: tuple[Leg, ...], snapshot: StrategySnapshot) -> float:
        # Sum mids: shorts contribute + (we receive), longs contribute - (we pay).
        # Mid for each leg = (bid+ask)/2 from the chain.
        chain_by_key = {
            (leg.expiry, leg.strike, leg.right): leg for leg in snapshot.chain
        }
        total = 0.0
        for leg in legs:
            # Skip stock or unsuitable legs whose option-key fields are None.
            if leg.expiry is None or leg.strike is None or leg.right is None:
                continue
            chain_leg = chain_by_key.get((leg.expiry, leg.strike, leg.right))
            if chain_leg is None or chain_leg.bid is None or chain_leg.ask is None:
                continue
            mid = (chain_leg.bid + chain_leg.ask) / 2
            total += mid if leg.side == "sell" else -mid
        return total * 100  # 1 contract = 100 shares

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # max loss = (max wing width) * 100 - credit
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
        # mypy: all four strikes are non-None per the guard above.
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
        # Iron condor max profit = credit received
        return self.estimate_credit(legs, snapshot)


