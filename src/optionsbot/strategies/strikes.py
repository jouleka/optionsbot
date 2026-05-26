"""Strike- and expiry-selection helpers reused by every strategy.

Pure functions over tuples of :class:`OptionChainLeg`. No I/O, no
randomness.
"""

from __future__ import annotations

from datetime import date, datetime

from optionsbot.ibkr.types import OptionChainLeg, OptionRight


def filter_by_expiry(
    chain: tuple[OptionChainLeg, ...], expiry: str
) -> tuple[OptionChainLeg, ...]:
    return tuple(leg for leg in chain if leg.expiry == expiry)


def filter_by_right(
    chain: tuple[OptionChainLeg, ...], right: OptionRight
) -> tuple[OptionChainLeg, ...]:
    return tuple(leg for leg in chain if leg.right == right)


def closest_expiry_to_dte(
    chain: tuple[OptionChainLeg, ...],
    dte_target: int,
    today: date | None = None,
) -> str | None:
    """Pick the chain expiry closest to ``dte_target`` calendar days from today."""
    today = today or date.today()
    expiries = sorted({leg.expiry for leg in chain})
    if not expiries:
        return None

    def dte(expiry: str) -> int:
        return (datetime.strptime(expiry, "%Y%m%d").date() - today).days

    return min(expiries, key=lambda e: abs(dte(e) - dte_target))


def closest_strike(
    legs: tuple[OptionChainLeg, ...], target_price: float
) -> OptionChainLeg | None:
    if not legs:
        return None
    return min(legs, key=lambda leg: abs(leg.strike - target_price))


def find_strike_by_delta(
    legs: tuple[OptionChainLeg, ...],
    target_delta: float,
    right: OptionRight,
) -> OptionChainLeg | None:
    """Find the leg whose delta is closest to ``target_delta``.

    For puts, the caller should pass a negative target (e.g., ``-0.16``).
    For calls, positive.
    """
    candidates = [leg for leg in legs if leg.right == right and leg.delta is not None]
    if not candidates:
        return None
    # mypy: candidates is filtered for non-None delta, but the lambda still
    # needs an explicit narrowing for strict mode.
    return min(
        candidates,
        key=lambda leg: abs((leg.delta if leg.delta is not None else 0.0) - target_delta),
    )
