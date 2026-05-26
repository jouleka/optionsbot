"""Calendar Spread (IBK-39) and Diagonal Spread (IBK-40).

Both are two-leg multi-expiry constructions. Differences:

* :class:`CalendarSpread` -- both legs are calls at the SAME (ATM) strike.
  Front-month leg is short, back-month leg is long. Profits from front-month
  theta decay outpacing the back-month's. Applicable to neutral direction
  with low or neutral IV.
* :class:`DiagonalSpread` -- two legs at DIFFERENT strikes AND different
  expiries. The ``right`` (call vs put) is selected by
  ``snapshot.view.direction`` so the same class covers both bullish and
  bearish biases. Applicable to bull/neutral/bear at low IV.

Both are defined-risk debit spreads. ``max_loss`` equals the absolute debit
paid; ``max_profit`` is left ``None`` since the back-leg outcome at front
expiry depends on where spot lands and how much time value remains in the
back leg -- not cleanly closed-form.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import ClassVar

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.ibkr.types import OptionChainLeg, OptionRight
from optionsbot.strategies.base import Leg, Strategy, StrategySnapshot
from optionsbot.strategies.strikes import (
    closest_strike,
    filter_by_expiry,
    filter_by_right,
)

# Calendar/diagonal-specific factor weights. Both share the same dict.
_CAL_DIAG_WEIGHTS: dict[str, float] = {
    "iv_rank": 0.15,
    "iv_hv": 0.10,
    "liquidity": 0.25,
    "dte_match": 0.15,
    "earnings_penalty": 0.20,
    "range_bound": 0.15,
}

# Minimum DTE for the front leg (skip "literally expires this week").
_MIN_FRONT_DTE = 14
# Minimum gap between front and back expiries.
_MIN_BACK_OVER_FRONT_DTE = 30


def _expiry_dte(expiry: str, today: date) -> int:
    return (datetime.strptime(expiry, "%Y%m%d").date() - today).days


def _pick_front_and_back_expiries(
    chain: tuple[OptionChainLeg, ...],
    today: date | None = None,
) -> tuple[str, str] | None:
    """Pick (front, back) expiries.

    Front = the nearest expiry with DTE >= ``_MIN_FRONT_DTE``.
    Back = the nearest expiry with DTE >= ``front_dte + _MIN_BACK_OVER_FRONT_DTE``.

    Returns ``None`` if either can't be found.
    """
    today = today or date.today()
    expiries = sorted({leg.expiry for leg in chain}, key=lambda e: _expiry_dte(e, today))
    front: str | None = None
    for exp in expiries:
        if _expiry_dte(exp, today) >= _MIN_FRONT_DTE:
            front = exp
            break
    if front is None:
        return None
    front_dte = _expiry_dte(front, today)
    back: str | None = None
    for exp in expiries:
        if _expiry_dte(exp, today) >= front_dte + _MIN_BACK_OVER_FRONT_DTE:
            back = exp
            break
    if back is None:
        return None
    return front, back


def _estimate_credit(
    legs: tuple[Leg, ...], snapshot: StrategySnapshot
) -> float:
    """Net credit (sells contribute +) per single contract set, in dollars.

    Shared with the straddle/vertical implementations; duplicated here to
    keep ``calendar.py`` self-contained.
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
# Calendar Spread (IBK-39)
# ---------------------------------------------------------------------------


class CalendarSpread(Strategy):
    name: ClassVar[str] = "calendar_spread"
    display_name: ClassVar[str] = "Calendar Spread"
    defined_risk: ClassVar[bool] = True
    # Calendar pays a debit but profits from theta on the front leg rather
    # than from long volatility, so it's NOT classified as long_premium for
    # IBK-5 scoring purposes.
    long_premium: ClassVar[bool] = False
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("neutral", "low"), ("neutral", "neutral")}
    )
    factor_weights: ClassVar[dict[str, float]] = _CAL_DIAG_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        pair = _pick_front_and_back_expiries(snapshot.chain)
        if pair is None:
            return None
        front_exp, back_exp = pair
        # Default to calls; calendars can be either side.
        right: OptionRight = "C"
        front_calls = filter_by_right(filter_by_expiry(snapshot.chain, front_exp), right)
        back_calls = filter_by_right(filter_by_expiry(snapshot.chain, back_exp), right)
        atm_front = closest_strike(front_calls, snapshot.spot)
        atm_back = closest_strike(back_calls, snapshot.spot)
        if atm_front is None or atm_back is None:
            return None
        # Pin both legs to the front-side strike for a clean same-strike
        # calendar; if the back side doesn't carry that strike, snap the
        # front to the back-side ATM instead.
        strike = atm_front.strike
        matching_back = next(
            (leg for leg in back_calls if leg.strike == strike), None
        )
        if matching_back is None:
            strike = atm_back.strike
            matching_front = next(
                (leg for leg in front_calls if leg.strike == strike), atm_front
            )
            front_strike = matching_front.strike
            back_strike = atm_back.strike
        else:
            front_strike = atm_front.strike
            back_strike = matching_back.strike
        return (
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=front_exp,
                strike=front_strike,
                right=right,
            ),
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=back_exp,
                strike=back_strike,
                right=right,
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
        # Calendar is a debit (credit < 0); max loss = absolute debit.
        return -credit if credit < 0 else 0.0


# ---------------------------------------------------------------------------
# Diagonal Spread (IBK-40)
# ---------------------------------------------------------------------------


class DiagonalSpread(Strategy):
    name: ClassVar[str] = "diagonal_spread"
    display_name: ClassVar[str] = "Diagonal Spread"
    defined_risk: ClassVar[bool] = True
    long_premium: ClassVar[bool] = False
    applicable_views: ClassVar[frozenset[tuple[Direction, IVRegime]]] = frozenset(
        {("bull", "low"), ("bear", "low"), ("neutral", "low")}
    )
    factor_weights: ClassVar[dict[str, float]] = _CAL_DIAG_WEIGHTS

    def suggest_legs(self, snapshot: StrategySnapshot) -> tuple[Leg, ...] | None:
        pair = _pick_front_and_back_expiries(snapshot.chain)
        if pair is None:
            return None
        front_exp, back_exp = pair
        # Calls for bull/neutral, puts for bear.
        right: OptionRight = "P" if snapshot.view.direction == "bear" else "C"
        front_side = filter_by_right(filter_by_expiry(snapshot.chain, front_exp), right)
        back_side = filter_by_right(filter_by_expiry(snapshot.chain, back_exp), right)
        if not front_side or not back_side:
            return None
        # The back leg sits at-the-money; the front leg sits one strike
        # further OTM (above spot for calls, below for puts). This gives
        # the diagonal its strike-stagger while keeping selection
        # deterministic when the chain doesn't carry distinct deltas.
        back_atm = closest_strike(back_side, snapshot.spot)
        if back_atm is None:
            return None
        if right == "C":
            otm_candidates = tuple(
                leg for leg in front_side if leg.strike > back_atm.strike
            )
        else:
            otm_candidates = tuple(
                leg for leg in front_side if leg.strike < back_atm.strike
            )
        if not otm_candidates:
            return None
        # Closest OTM strike beyond the back-leg strike.
        front_leg = min(
            otm_candidates, key=lambda leg: abs(leg.strike - back_atm.strike)
        )
        return (
            Leg(
                symbol=snapshot.symbol,
                side="sell",
                expiry=front_exp,
                strike=front_leg.strike,
                right=right,
            ),
            Leg(
                symbol=snapshot.symbol,
                side="buy",
                expiry=back_exp,
                strike=back_atm.strike,
                right=right,
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
