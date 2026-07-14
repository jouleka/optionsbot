"""Broker-independent structural validation for persisted option legs."""

from __future__ import annotations

import math
from collections import defaultdict


def has_structurally_defined_option_risk(legs: object) -> bool:
    """Return True only when every persisted short option is fully covered."""
    if not isinstance(legs, list) or not legs:
        return False
    quantities: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"buy": 0, "sell": 0}
    )
    for raw_leg in legs:
        if not isinstance(raw_leg, dict) or raw_leg.get("sec_type") != "OPT":
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
        if not math.isfinite(float(strike)) or float(strike) <= 0:
            return False
        quantity = raw_leg.get("quantity")
        if type(quantity) is not int or quantity <= 0:
            return False
        quantities[(symbol.strip().upper(), expiry, right)][side] += quantity
    return all(
        totals["sell"] == 0 or totals["buy"] >= totals["sell"]
        for totals in quantities.values()
    )
