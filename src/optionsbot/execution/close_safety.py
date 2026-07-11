"""Close-safety guards (Phase 0 C2).

Two independent checks protect autonomous closes:

1. ``assert_atomic_close_legs`` — BEFORE placing a close, verify it is the
   exact side-flipped inverse of the entry's option legs. ``place_combo_limit``
   routes any multi-leg structure as one guaranteed SMART BAG (atomic), but
   only if it is actually handed every leg. A close that lost a leg (would
   route as a bare single Option) or that is not this position's inverse is
   refused — we fail safe and halt rather than leg out and strand a naked side.

2. ``find_naked_short_legs`` — AFTER a close fills, compare the broker's actual
   per-leg positions to the entry. Any SHORT option leg still open at the
   broker (position < 0) is a residual naked short — a P1 incident. A residual
   LONG leg is defined risk (capital already spent) and is not flagged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from optionsbot.ibkr.types import PortfolioPosition

LegSpec = tuple[str, float, str]  # (expiry, strike, right)


class NonAtomicCloseError(RuntimeError):
    """The staged close cannot be guaranteed to execute as one atomic combo
    against this position — placing it could leg out into a naked short."""


def _option_legs(legs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [leg for leg in legs if leg.get("sec_type", "OPT") == "OPT"]


def _spec(leg: Mapping[str, Any]) -> LegSpec:
    return (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))


def _flip(side: str) -> str:
    if side == "sell":
        return "buy"
    if side == "buy":
        return "sell"
    raise NonAtomicCloseError(f"unsupported option leg side {side!r}")


def _validated_leg_map(
    legs: Sequence[Mapping[str, Any]], *, label: str, flip_sides: bool
) -> dict[tuple[str, str, float, str], tuple[str, int]]:
    normalized: dict[tuple[str, str, float, str], tuple[str, int]] = {}
    for leg in legs:
        try:
            symbol = leg["symbol"]
            side = leg["side"]
            raw_quantity = leg.get("quantity", 1)
            quantity = int(raw_quantity)
            spec = _spec(leg)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise NonAtomicCloseError(f"{label} contains a malformed option leg") from exc
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
            or not isinstance(side, str)
            or side not in {"buy", "sell"}
            or isinstance(raw_quantity, bool)
            or quantity <= 0
            or quantity != raw_quantity
        ):
            raise NonAtomicCloseError(f"{label} contains a malformed option leg")
        identity = (symbol.strip().upper(), *spec)
        if identity in normalized:
            raise NonAtomicCloseError(
                f"{label} contains duplicate option contract identity {identity}"
            )
        normalized[identity] = (_flip(side) if flip_sides else side, quantity)
    return normalized


def assert_atomic_close_legs(
    *,
    entry_legs: Sequence[Mapping[str, Any]],
    close_legs: Sequence[Mapping[str, Any]],
) -> None:
    """Raise ``NonAtomicCloseError`` unless ``close_legs`` is exactly the
    side-flipped inverse of the entry's option legs (same set of
    expiry/strike/right, every side reversed, same per-leg quantity)."""
    entry_opts = _option_legs(entry_legs)
    close_opts = _option_legs(close_legs)
    if not entry_opts:
        raise NonAtomicCloseError("entry has no option legs to close")
    if len(close_opts) != len(entry_opts):
        raise NonAtomicCloseError(
            f"close has {len(close_opts)} option legs, entry has "
            f"{len(entry_opts)} — cannot close atomically"
        )
    expected = _validated_leg_map(entry_opts, label="entry", flip_sides=True)
    actual = _validated_leg_map(close_opts, label="close", flip_sides=False)
    if expected != actual:
        raise NonAtomicCloseError(
            f"close legs are not the inverse of the entry "
            f"(expected {expected}, got {actual})"
        )


def find_naked_short_legs(
    entry_legs: Sequence[Mapping[str, Any]],
    broker_positions: Sequence[PortfolioPosition],
) -> list[PortfolioPosition]:
    """Return broker positions that are a residual NAKED SHORT of one of this
    entry's legs (sold leg still open, position < 0) after a close.

    Matched by (expiry, strike, right) on OPT positions in the same symbol.
    Only short (negative) residuals are returned; long residuals are not
    naked-short risk."""
    entry_specs = {_spec(leg) for leg in _option_legs(entry_legs)}
    symbols = {str(leg["symbol"]) for leg in _option_legs(entry_legs)}
    naked: list[PortfolioPosition] = []
    for pos in broker_positions:
        if pos.sec_type != "OPT" or pos.symbol not in symbols:
            continue
        if pos.expiry is None or pos.strike is None or pos.right is None:
            continue
        spec = (str(pos.expiry), float(pos.strike), str(pos.right))
        if spec in entry_specs and pos.position < 0:
            naked.append(pos)
    return naked
