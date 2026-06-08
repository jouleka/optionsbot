"""Numeric-cleanup helpers shared across the IBKR adapter layer.

IBKR's ib_async returns NaN for missing numeric fields (bid, ask, IV,
greeks, open_interest, volume, etc.). Downstream code uniformly wants
None for missing values so adapter dataclasses and SQLite columns stay
clean. These two helpers centralize the NaN -> None conversion.
"""

from __future__ import annotations

import math

# IBKR sends sys.float_info.max (~1.7977e308) as the "unset double" sentinel for
# numeric fields it has no value for (e.g. unrealizedPNL on a not-yet-priced option
# leg). ib_async's portfolio decoder passes it through as a raw float rather than
# converting it, so treat it -- and any non-finite value -- as missing, alongside NaN.
_UNSET_DOUBLE_THRESHOLD = 1e308


def clean_float(value: float | None) -> float | None:
    """Return value, or None when it is None, NaN/inf, or IBKR's unset-double sentinel."""
    if value is None:
        return None
    try:
        if not math.isfinite(value) or abs(value) >= _UNSET_DOUBLE_THRESHOLD:
            return None
    except TypeError:
        return value
    return value


def clean_int(value: float | int | None) -> int | None:
    """Return int(value), or None when value is None, NaN, or unconvertible."""
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        # Not a float-like (e.g., already int); fall through to int() below.
        pass
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(math.inf) raises; treat as missing per the
        # function's documented "or unconvertible -> None" contract.
        return None
