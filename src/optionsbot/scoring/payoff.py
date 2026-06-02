"""Expiry-payoff + probability-of-profit math (IBK-93).

Pure stdlib. A strategy's profit/loss at expiry is a deterministic function of
the underlying price; integrating that over a lognormal price-at-expiry
distribution gives the probability of finishing profitable.

Only single-expiry, all-option positions are modelable here: stock legs and
multi-expiry (calendar/diagonal) positions are path-dependent and return None.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from optionsbot.strategies.base import Leg

# Integration grid over the standard-normal z in [-_Z, +_Z]: _STEPS intervals =>
# _STEPS + 1 sample points (endpoints included; phi(+-5) ~ 3.7e-6 so the missing
# trapezoidal half-weight is negligible). Accurate to ~1e-3 vs the analytical CDF
# at realistic scanner inputs (IV <= ~2.0, DTE <= ~90). Long-dated/extreme-IV
# inputs (sigma > ~1) would need a wider grid; out of scope for this scanner.
_Z = 5.0
_STEPS = 1000


def is_terminal_modelable(legs: Iterable[Leg]) -> bool:
    """True iff every leg is an option (no stock) and all share ONE expiry.

    Calendars/diagonals (multiple expiries) and stock-leg strategies are
    path-dependent at the front expiry, so a single terminal-price model
    can't value them.
    """
    legs = tuple(legs)
    if not legs:
        return False
    expiries: set[str] = set()
    for leg in legs:
        if leg.sec_type != "OPT" or leg.strike is None or leg.right is None or leg.expiry is None:
            return False
        expiries.add(leg.expiry)
    return len(expiries) == 1


def terminal_pnl_dollars(
    legs: Iterable[Leg], credit_or_debit: float, s_t: float
) -> float:
    """P&L in dollars (per single contract set) if the underlying = ``s_t`` at expiry.

    P&L = sum(sign * intrinsic * 100 * qty) + entry cashflow, where sign is +1
    for long (buy) and -1 for short (sell), and ``credit_or_debit`` is the entry
    cashflow (positive credit / negative debit).

    PRECONDITION: option legs only. Stock legs (``sec_type != "OPT"``) are SKIPPED,
    so a position containing one yields a WRONG (partial) result. Callers MUST gate
    with ``is_terminal_modelable`` first (``prob_of_profit`` already does).
    """
    total = credit_or_debit
    for leg in legs:
        if leg.strike is None or leg.right is None:
            continue
        if leg.right == "C":
            intrinsic = max(0.0, s_t - leg.strike)
        else:
            intrinsic = max(0.0, leg.strike - s_t)
        sign = 1.0 if leg.side == "buy" else -1.0
        total += sign * intrinsic * 100.0 * leg.quantity
    return total


def prob_of_profit(
    legs: Iterable[Leg],
    credit_or_debit: float,
    spot: float,
    atm_iv: float | None,
    dte_days: float,
) -> float | None:
    """P(P&L at expiry > 0) under a lognormal terminal-price distribution.

    Returns None when not modelable (stock/multi-expiry legs) or when inputs are
    missing/degenerate (no IV, non-positive spot/DTE).
    """
    legs = tuple(legs)
    if atm_iv is None or atm_iv <= 0.0 or spot <= 0.0 or dte_days <= 0.0:
        return None
    if not is_terminal_modelable(legs):
        return None
    sigma = atm_iv * math.sqrt(dte_days / 365.0)
    if sigma <= 0.0:
        return None
    # Risk-neutral, zero-drift lognormal: S_T = spot * exp(-0.5 sigma^2 + sigma z),
    # z ~ N(0,1). Weight each grid point by the (unnormalized) standard-normal pdf;
    # normalization cancels in the ratio.
    step = (2.0 * _Z) / _STEPS
    w_total = 0.0
    w_profit = 0.0
    for i in range(_STEPS + 1):
        z = -_Z + i * step
        phi = math.exp(-0.5 * z * z)
        s_t = spot * math.exp(-0.5 * sigma * sigma + sigma * z)
        w_total += phi
        if terminal_pnl_dollars(legs, credit_or_debit, s_t) > 0.0:
            w_profit += phi
    if w_total <= 0.0:
        return None
    return w_profit / w_total
