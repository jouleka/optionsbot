"""Numeric-cleanup helpers shared across the IBKR adapter layer.

IBKR's ib_async returns NaN for missing numeric fields (bid, ask, IV,
greeks, open_interest, volume, etc.). Downstream code uniformly wants
None for missing values so adapter dataclasses and SQLite columns stay
clean. These two helpers centralize the NaN -> None conversion.
"""

from __future__ import annotations

import math


def clean_float(value: float | None) -> float | None:
    """Return value, or None when it is None or NaN."""
    if value is None:
        return None
    try:
        if math.isnan(value):
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
    except (TypeError, ValueError):
        return None
