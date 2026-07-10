"""Conservative structural validation for option entry legs.

A persisted ``defined_risk`` flag is advisory only. Execution requires the
legs themselves to prove that every short option group is fully covered by
long options of the same underlying, expiry, and right. This deliberately
rejects calendars and other structures whose bounded loss cannot be proven
from the compact persisted leg packet.
"""

from __future__ import annotations

import math
from collections import defaultdict


def has_structurally_defined_option_risk(legs: object) -> bool:
    """Return True only when the persisted option legs prove bounded short risk."""
    if not isinstance(legs, list) or not legs:
        return False

    quantities: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"buy": 0, "sell": 0}
    )
    for raw_leg in legs:
        if not isinstance(raw_leg, dict):
            return False
        if raw_leg.get("sec_type") != "OPT":
            return False
        side = raw_leg.get("side")
        if side not in {"buy", "sell"}:
            return False
        symbol = raw_leg.get("symbol")
        expiry = raw_leg.get("expiry")
        right = raw_leg.get("right")
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        if not isinstance(expiry, str) or len(expiry) != 8 or not expiry.isdigit():
            return False
        if right not in {"C", "P"}:
            return False
        strike = raw_leg.get("strike")
        if isinstance(strike, bool) or not isinstance(strike, (int, float)):
            return False
        strike_value = float(strike)
        if not math.isfinite(strike_value) or strike_value <= 0:
            return False
        quantity = raw_leg.get("quantity")
        if type(quantity) is not int or quantity <= 0:  # bool must not count as 1
            return False
        quantities[(symbol.strip().upper(), expiry, right)][side] += quantity

    return all(
        totals["sell"] == 0 or totals["buy"] >= totals["sell"]
        for totals in quantities.values()
    )
