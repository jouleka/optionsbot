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


def structural_max_loss_dollars(
    legs: object,
    *,
    entry_net_per_share: float,
) -> float | None:
    """Return a conservative expiry max loss derived from legs and entry price.

    ``entry_net_per_share`` is positive for a credit and negative for a debit.
    Independent expiry/underlying groups are allowed, but their individual worst
    intrinsic outcomes are summed so the result never assumes favorable
    cross-expiry price correlation. ``None`` means the bound cannot be proven.
    """
    if (
        not has_structurally_defined_option_risk(legs)
        or not math.isfinite(entry_net_per_share)
    ):
        return None
    assert isinstance(legs, list)

    groups: dict[tuple[str, str], list[tuple[str, str, float, int]]] = defaultdict(list)
    identities: set[tuple[str, str, str, float]] = set()
    for leg in legs:
        symbol = str(leg["symbol"]).strip().upper()
        expiry = str(leg["expiry"])
        right = str(leg["right"])
        side = str(leg["side"])
        strike = float(leg["strike"])
        quantity = int(leg["quantity"])
        identity = (symbol, expiry, right, strike)
        if identity in identities:
            return None
        identities.add(identity)
        groups[(symbol, expiry)].append((side, right, strike, quantity))

    worst_intrinsic_per_share = 0.0
    for group_legs in groups.values():
        # A net short call slope above every strike is unbounded.
        high_price_slope = sum(
            (1 if side == "buy" else -1) * quantity
            for side, right, _strike, quantity in group_legs
            if right == "C"
        )
        if high_price_slope < 0:
            return None
        checkpoints = {0.0, *(strike for _side, _right, strike, _qty in group_legs)}
        group_min = math.inf
        for spot in checkpoints:
            intrinsic = 0.0
            for side, right, strike, quantity in group_legs:
                payoff = (
                    max(spot - strike, 0.0)
                    if right == "C"
                    else max(strike - spot, 0.0)
                )
                intrinsic += (1.0 if side == "buy" else -1.0) * quantity * payoff
            group_min = min(group_min, intrinsic)
        if not math.isfinite(group_min):
            return None
        worst_intrinsic_per_share += group_min

    worst_pnl = (entry_net_per_share + worst_intrinsic_per_share) * 100.0
    if not math.isfinite(worst_pnl):
        return None
    return max(0.0, -worst_pnl)


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
