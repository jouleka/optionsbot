"""Dynamic position sizing (IBK-133): how professionals size, bounded.

Computed at EXECUTE time from live equity + the bot's own realized history:
base risk × edge tilt (quarter-Kelly from the pick's own PoP and payoff,
clamped ×0.5–×2.0) × drawdown governor (anti-martingale: NEVER sizes up
after losses), then capped by portfolio heat and a single-trade ceiling.
Small accounts get a minimum-viable 1 lot when the trade fits the caps —
fractional contracts don't exist.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, select

from optionsbot.storage.schema import fills, orders, strategy_scores


@dataclass(frozen=True, slots=True)
class SizeDecision:
    quantity: int
    note: str  # the working, shown in the /execute reply


def _scoreless_order_max_loss(engine: Engine, row: Any) -> float:
    """Derive max loss for a fully filled, single-expiry adopted structure.

    Scoreless broker adoptions have no persisted scoring packet. Their expiry
    payoff is usable only when every expected leg fill is present and finite.
    Unknown, mixed-expiry, malformed, partially filled, or unbounded risk fails
    closed as infinity so the sizing gate cannot add exposure.
    """
    legs = list(row.legs_json or [])
    if not legs or any(leg.get("sec_type", "OPT") != "OPT" for leg in legs):
        return math.inf
    raw_order_quantity = row.quantity
    try:
        order_quantity = int(raw_order_quantity)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if (
        isinstance(raw_order_quantity, bool)
        or order_quantity <= 0
        or order_quantity != raw_order_quantity
    ):
        return math.inf

    normalized: list[tuple[str, str, float, int]] = []
    expiries: set[str] = set()
    underlyings: set[str] = set()
    expected_fill_qty = {"BUY": 0, "SELL": 0}
    leg_count_by_side = {"BUY": 0, "SELL": 0}
    expected_by_con_id: dict[int, tuple[str, int]] = {}
    all_legs_have_con_id = True
    high_spot_call_slope = 0
    for leg in legs:
        side = str(leg.get("side", "")).lower()
        right = str(leg.get("right", "")).upper()
        expiry = str(leg.get("expiry", ""))
        underlying = str(leg.get("symbol", "")).upper()
        try:
            datetime.strptime(expiry, "%Y%m%d")
        except ValueError:
            return math.inf
        try:
            strike = float(leg["strike"])
            raw_leg_quantity = leg.get("quantity", 1)
            leg_ratio = int(raw_leg_quantity)
            raw_multiplier = leg["multiplier"]
            multiplier = int(raw_multiplier)
            leg_quantity = leg_ratio * order_quantity
        except (KeyError, TypeError, ValueError, OverflowError):
            return math.inf
        if (
            side not in {"buy", "sell"}
            or right not in {"C", "P"}
            or len(expiry) != 8
            or not expiry.isdigit()
            or not math.isfinite(strike)
            or strike <= 0
            or leg.get("currency") != "USD"
            or isinstance(raw_multiplier, bool)
            or multiplier != 100
            or multiplier != raw_multiplier
            or isinstance(raw_leg_quantity, bool)
            or leg_ratio <= 0
            or leg_ratio != raw_leg_quantity
        ):
            return math.inf
        sign = 1 if side == "buy" else -1
        fill_side = "BUY" if side == "buy" else "SELL"
        raw_con_id = leg.get("con_id")
        if raw_con_id is None:
            all_legs_have_con_id = False
        else:
            try:
                con_id = int(raw_con_id)
            except (TypeError, ValueError, OverflowError):
                return math.inf
            if (
                isinstance(raw_con_id, bool)
                or con_id <= 0
                or con_id != raw_con_id
                or con_id in expected_by_con_id
            ):
                return math.inf
            expected_by_con_id[con_id] = (fill_side, leg_quantity)
        if right == "C":
            high_spot_call_slope += sign * leg_quantity
        expiries.add(expiry)
        underlyings.add(underlying)
        expected_fill_qty[fill_side] += leg_quantity
        leg_count_by_side[fill_side] += 1
        normalized.append((side, right, strike, leg_quantity))
    if (
        underlyings != {str(row.symbol).upper()}
        or len(expiries) != 1
        or high_spot_call_slope < 0
        or (not all_legs_have_con_id and bool(expected_by_con_id))
        or (not all_legs_have_con_id and any(count > 1 for count in leg_count_by_side.values()))
    ):
        return math.inf

    with engine.connect() as conn:
        fill_rows = conn.execute(
            select(fills.c.side, fills.c.price, fills.c.qty, fills.c.leg_con_id).where(
                fills.c.order_id == int(row.id)
            )
        ).fetchall()
    if not fill_rows:
        return math.inf
    actual_fill_qty = {"BUY": 0, "SELL": 0}
    actual_by_con_id: dict[int, tuple[str, int]] = {}
    premium = 0.0
    for fill in fill_rows:
        try:
            price = float(fill.price)
            qty = int(fill.qty)
        except (TypeError, ValueError):
            return math.inf
        if (
            fill.side not in actual_fill_qty
            or not math.isfinite(price)
            or price < 0
            or qty <= 0
            or qty != fill.qty
        ):
            return math.inf
        actual_fill_qty[fill.side] += qty
        if all_legs_have_con_id:
            if fill.leg_con_id is None:
                return math.inf
            con_id = int(fill.leg_con_id)
            expected = expected_by_con_id.get(con_id)
            if expected is None or expected[0] != fill.side:
                return math.inf
            prior = actual_by_con_id.get(con_id, (fill.side, 0))
            actual_by_con_id[con_id] = (fill.side, prior[1] + qty)
        premium += (1.0 if fill.side == "SELL" else -1.0) * price * qty * 100.0
    if (
        actual_fill_qty != expected_fill_qty
        or (all_legs_have_con_id and actual_by_con_id != expected_by_con_id)
        or not math.isfinite(premium)
    ):
        return math.inf

    spots = {0.0, *(strike for _, _, strike, _ in normalized)}
    worst_pnl = math.inf
    for spot in spots:
        pnl = premium
        for side, right, strike, quantity in normalized:
            intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
            pnl += (1.0 if side == "buy" else -1.0) * intrinsic * quantity * 100.0
        worst_pnl = min(worst_pnl, pnl)
    max_loss = max(0.0, -worst_pnl)
    return max_loss if math.isfinite(max_loss) else math.inf


def open_heat_dollars(engine: Engine) -> float:
    """Total max-loss of all open bot and adopted positions.

    Scored bot entries use their persisted ``max_loss``. Scoreless/manual
    adoptions are derived from their actual fills and option payoff. Unknown or
    unbounded adopted risk returns ``inf`` so sizing refuses additional risk.
    """
    active = ("staged", "submitting", "submitted", "partial", "filled")
    with engine.connect() as conn:
        closed_ids = {
            row.closes_order_id
            for row in conn.execute(
                select(orders.c.closes_order_id)
                .where(orders.c.intent == "close")
                .where(orders.c.status == "filled")
            ).fetchall()
        }
        rows = conn.execute(
            select(
                orders.c.id,
                orders.c.symbol,
                orders.c.quantity,
                orders.c.legs_json,
                orders.c.limit_price,
                strategy_scores.c.suggestion_json,
            )
            .outerjoin(strategy_scores, orders.c.strategy_score_id == strategy_scores.c.id)
            .where(orders.c.intent == "open")
            .where(orders.c.status.in_(active))
        ).fetchall()
    heat = 0.0
    for row in rows:
        if row.id in closed_ids:
            continue
        suggestion = row.suggestion_json
        if isinstance(suggestion, str):  # defensive: JSON column round-trip
            try:
                suggestion = json.loads(suggestion)
            except (TypeError, ValueError):
                return math.inf
        if suggestion is not None and not isinstance(suggestion, dict):
            return math.inf
        max_loss = (suggestion or {}).get("max_loss")
        if max_loss is not None:
            try:
                scored_loss = float(max_loss)
                quantity = int(row.quantity)
            except (TypeError, ValueError, OverflowError):
                return math.inf
            if not math.isfinite(scored_loss) or scored_loss <= 0 or quantity <= 0:
                return math.inf
            heat += scored_loss * quantity
            if not math.isfinite(heat):
                return math.inf
            continue
        derived = _scoreless_order_max_loss(engine, row)
        if not math.isfinite(derived):
            return math.inf
        heat += derived
    return heat


def _loss_streak(recent_pnls: list[float]) -> int:
    streak = 0
    for pnl in reversed(recent_pnls):
        if pnl < 0:
            streak += 1
        else:
            break
    return streak


def dynamic_quantity(
    *,
    equity: float,
    max_loss_unit: float,  # dollars per contract set
    max_profit_unit: float | None,
    prob_profit: float | None,
    open_heat: float,
    recent_pnls: list[float],
    base_risk_pct: float,
    heat_cap_pct: float,
    single_trade_cap_pct: float,
) -> SizeDecision:
    if equity <= 0 or max_loss_unit <= 0:
        return SizeDecision(0, "no equity/defined risk basis")

    base = equity * base_risk_pct

    # Edge tilt: quarter-Kelly from the pick's own numbers, bounded.
    tilt = 0.5
    if max_profit_unit and prob_profit and max_profit_unit > 0 and 0 < prob_profit < 1:
        b = max_profit_unit / max_loss_unit
        kelly = prob_profit - (1 - prob_profit) / b
        if kelly > 0:
            tilt = max(0.5, min(2.0, (equity * kelly / 4) / base))

    # Drawdown governor — anti-martingale only.
    streak = _loss_streak(recent_pnls)
    governor = 0.5 if streak >= 3 else (0.7 if streak == 2 else 1.0)

    budget = base * tilt * governor

    single_cap = equity * single_trade_cap_pct
    if max_loss_unit > single_cap:
        return SizeDecision(
            0,
            f"max loss ${max_loss_unit:,.0f}/contract exceeds the "
            f"{single_trade_cap_pct * 100:.0f}% single-trade cap "
            f"(${single_cap:,.0f}) on ${equity:,.0f} equity",
        )
    heat_room = equity * heat_cap_pct - open_heat
    if heat_room < max_loss_unit:
        return SizeDecision(
            0,
            f"portfolio heat ${open_heat:,.0f} leaves no room under the "
            f"{heat_cap_pct * 100:.0f}% cap",
        )

    quantity = math.floor(min(budget, single_cap, heat_room) / max_loss_unit)
    floored = ""
    if quantity == 0:
        if streak >= 3:
            # Drawdown governor is active: a losing streak that sizes to 0
            # must SKIP the trade, not get floored up to a min-1 lot.
            return SizeDecision(
                0,
                f"skipped — loss streak {streak} (governor ×{governor:.1f}) "
                f"sizes below 1 lot; no min-1 floor while drawing down",
            )
        quantity = 1  # minimum viable lot — it already fits both caps
        floored = ", min-1"
    note = (
        f"sized {quantity}x — base ${base:,.0f}, edge ×{tilt:.1f}, "
        f"dd ×{governor:.1f}{floored}; heat ${open_heat + max_loss_unit * quantity:,.0f}"
        f"/${equity * heat_cap_pct:,.0f}"
    )
    return SizeDecision(quantity, note)
