"""Vertical spreads: Bull Put / Bear Call (credit) + Bull Call / Bear Put (debit).

Every vertical is two same-expiry legs of the same right:

* Credit spreads (Bull Put, Bear Call) sell a higher-premium (closer-to-ATM)
  option at ~0.30 delta and buy a cheaper protective long ~5% further OTM.
* Debit spreads (Bull Call, Bear Put) buy a near-ATM (~0.50 delta) option
  and sell a further-OTM (~0.20 delta) option of the same right, paying
  the difference.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.ibkr.types import OptionChainLeg, OptionRight
from optionsbot.strategies.base import Leg, Strategy, StrategySnapshot
from optionsbot.strategies.strikes import (
    closest_expiry_to_dte,
    filter_by_expiry,
    filter_by_right,
    find_strike_by_delta,
)

_CREDIT_SHORT_DELTA = 0.30  # magnitude; sign supplied per side
_CREDIT_LONG_OFFSET_PCT = 0.05  # 5% of spot beyond the short strike
_DEBIT_LONG_DELTA = 0.50  # magnitude
_DEBIT_SHORT_DELTA = 0.20  # magnitude


def _spread_legs(
    snapshot: StrategySnapshot,
    right: OptionRight,
    short_delta_target: float,
    long_delta_target: float | None,
    long_offset_pct: float | None,
    direction: Literal["up", "down"],
) -> tuple[Leg, Leg] | None:
    """Build the two legs of a vertical spread.

    ``right`` is "P" for put spreads, "C" for call spreads. ``direction``
    is the *strike* direction the long leg sits relative to the short:
    "up" means the long strike is higher than the short, "down" means lower.

    Exactly one of ``long_delta_target`` or ``long_offset_pct`` should be
    provided. Credit spreads use the offset (chain-distance hedge); debit
    spreads use a delta target (the long is ATM-ish and the short further OTM
    -- here the "short" is the smaller-delta one and the "long" is the larger).
    """
    expiry = closest_expiry_to_dte(snapshot.chain, snapshot.dte_target)
    if expiry is None:
        return None
    legs_at_expiry = filter_by_expiry(snapshot.chain, expiry)
    side_legs = filter_by_right(legs_at_expiry, right)
    short_leg = find_strike_by_delta(side_legs, short_delta_target, right)
    if short_leg is None:
        return None
    long_leg: OptionChainLeg | None
    if long_offset_pct is not None:
        # Credit spread: long strike sits ``long_offset_pct`` beyond the short
        # strike in the indicated direction.
        offset = max(1.0, snapshot.spot * long_offset_pct)
        if direction == "down":
            long_target_price = short_leg.strike - offset
            candidates = [leg for leg in side_legs if leg.strike < short_leg.strike]
        else:
            long_target_price = short_leg.strike + offset
            candidates = [leg for leg in side_legs if leg.strike > short_leg.strike]
        long_leg = min(
            candidates,
            key=lambda leg: abs(leg.strike - long_target_price),
            default=None,
        )
    else:
        # Debit spread: pick the long by delta target. Constrain to the side
        # of the short strike that makes the spread a vertical (long lower for
        # bull call, long higher for bear put).
        assert long_delta_target is not None
        if direction == "down":
            # Long is below short (bear put: long higher-strike put, short
            # lower-strike put -> wait: direction "down" means long sits BELOW
            # short. Bear put: short is below long, so we never reach here for
            # bear put). This branch is used by bull call: short above long, so
            # the long lives at *lower* strikes than the short.
            candidates = [leg for leg in side_legs if leg.strike < short_leg.strike]
        else:
            candidates = [leg for leg in side_legs if leg.strike > short_leg.strike]
        long_leg = min(
            candidates,
            key=lambda leg: abs(
                (leg.delta if leg.delta is not None else 0.0) - long_delta_target
            ),
            default=None,
        )
    if long_leg is None:
        return None
    short = Leg(
        symbol=snapshot.symbol,
        side="sell",
        expiry=expiry,
        strike=short_leg.strike,
        right=right,
    )
    long_ = Leg(
        symbol=snapshot.symbol,
        side="buy",
        expiry=expiry,
        strike=long_leg.strike,
        right=right,
    )
    return short, long_


def _estimate_credit(legs: tuple[Leg, ...], snapshot: StrategySnapshot) -> float:
    """Net credit/debit per single contract (× 100). Shared by all 4 verticals."""
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


_CREDIT_WEIGHTS: dict[str, float] = {
    "iv_rank": 0.30,
    "iv_hv": 0.20,
    "liquidity": 0.20,
    "dte_match": 0.10,
    "earnings_penalty": 0.10,
    "range_bound": 0.10,
}


_DEBIT_WEIGHTS: dict[str, float] = {
    "iv_rank": 0.10,
    "iv_hv": 0.15,
    "liquidity": 0.20,
    "dte_match": 0.20,
    "earnings_penalty": 0.15,
    "range_bound": 0.20,
}


# ---------------------------------------------------------------------------
# Bull Put Spread (credit)
# ---------------------------------------------------------------------------


class BullPutSpread(Strategy):
    name: ClassVar[str] = "bull_put_spread"
    display_name: ClassVar[str] = "Bull Put Spread"
    defined_risk: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bull", "high"), ("bull", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _CREDIT_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        # Short put at ~-0.30 delta, long put ~5% below short.
        pair = _spread_legs(
            snapshot,
            right="P",
            short_delta_target=-_CREDIT_SHORT_DELTA,
            long_delta_target=None,
            long_offset_pct=_CREDIT_LONG_OFFSET_PCT,
            direction="down",
        )
        return pair

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        short = next((leg for leg in legs if leg.side == "sell"), None)
        long_ = next((leg for leg in legs if leg.side == "buy"), None)
        if short is None or long_ is None:
            return None
        if short.strike is None or long_.strike is None:
            return None
        width = short.strike - long_.strike
        return width * 100 - self.estimate_credit(legs, snapshot)

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        return self.estimate_credit(legs, snapshot)


# ---------------------------------------------------------------------------
# Bear Call Spread (credit)
# ---------------------------------------------------------------------------


class BearCallSpread(Strategy):
    name: ClassVar[str] = "bear_call_spread"
    display_name: ClassVar[str] = "Bear Call Spread"
    defined_risk: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bear", "high"), ("bear", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _CREDIT_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        # Short call at ~+0.30 delta, long call ~5% above short.
        return _spread_legs(
            snapshot,
            right="C",
            short_delta_target=_CREDIT_SHORT_DELTA,
            long_delta_target=None,
            long_offset_pct=_CREDIT_LONG_OFFSET_PCT,
            direction="up",
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        short = next((leg for leg in legs if leg.side == "sell"), None)
        long_ = next((leg for leg in legs if leg.side == "buy"), None)
        if short is None or long_ is None:
            return None
        if short.strike is None or long_.strike is None:
            return None
        width = long_.strike - short.strike
        return width * 100 - self.estimate_credit(legs, snapshot)

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        return self.estimate_credit(legs, snapshot)


# ---------------------------------------------------------------------------
# Bull Call Spread (debit)
# ---------------------------------------------------------------------------


class BullCallSpread(Strategy):
    name: ClassVar[str] = "bull_call_spread"
    display_name: ClassVar[str] = "Bull Call Spread"
    defined_risk: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bull", "low"), ("bull", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _DEBIT_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        # Long call ATM-ish (~+0.50 delta), short call further OTM (~+0.20).
        # Anchor on the short (smaller-delta) leg via the shared helper, then
        # the helper picks the long at ``long_delta_target`` on the strike side
        # below the short.
        return _spread_legs(
            snapshot,
            right="C",
            short_delta_target=_DEBIT_SHORT_DELTA,
            long_delta_target=_DEBIT_LONG_DELTA,
            long_offset_pct=None,
            direction="down",
        )

    def estimate_credit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float:
        return _estimate_credit(legs, snapshot)

    def estimate_max_loss(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        # Debit spread max loss = absolute debit paid (which is -credit since
        # credit is negative for a debit position).
        credit = self.estimate_credit(legs, snapshot)
        return -credit if credit < 0 else 0.0

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        short = next((leg for leg in legs if leg.side == "sell"), None)
        long_ = next((leg for leg in legs if leg.side == "buy"), None)
        if short is None or long_ is None:
            return None
        if short.strike is None or long_.strike is None:
            return None
        width = short.strike - long_.strike
        debit = -self.estimate_credit(legs, snapshot)
        return width * 100 - debit


# ---------------------------------------------------------------------------
# Bear Put Spread (debit)
# ---------------------------------------------------------------------------


class BearPutSpread(Strategy):
    name: ClassVar[str] = "bear_put_spread"
    display_name: ClassVar[str] = "Bear Put Spread"
    defined_risk: ClassVar[bool] = True
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bear", "low"), ("bear", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _DEBIT_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        # Long put ATM-ish (~-0.50 delta), short put further OTM (~-0.20).
        # Anchor on the short (smaller-magnitude-delta) leg; the helper picks
        # the long at the larger-magnitude delta on the strike side above the
        # short (direction="up" for puts means long is at HIGHER strikes).
        return _spread_legs(
            snapshot,
            right="P",
            short_delta_target=-_DEBIT_SHORT_DELTA,
            long_delta_target=-_DEBIT_LONG_DELTA,
            long_offset_pct=None,
            direction="up",
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

    def estimate_max_profit(
        self, legs: tuple[Leg, ...], snapshot: StrategySnapshot
    ) -> float | None:
        short = next((leg for leg in legs if leg.side == "sell"), None)
        long_ = next((leg for leg in legs if leg.side == "buy"), None)
        if short is None or long_ is None:
            return None
        if short.strike is None or long_.strike is None:
            return None
        width = long_.strike - short.strike
        debit = -self.estimate_credit(legs, snapshot)
        return width * 100 - debit
