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

from sqlalchemy import Engine, select

from optionsbot.storage.schema import orders, strategy_scores


@dataclass(frozen=True, slots=True)
class SizeDecision:
    quantity: int
    note: str  # the working, shown in the /execute reply


def open_heat_dollars(engine: Engine) -> float:
    """Σ max-loss of the bot's OPEN positions (active entries minus completed
    round-trips), from each pick's persisted suggestion."""
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
            select(orders.c.id, orders.c.quantity, strategy_scores.c.suggestion_json)
            .join(strategy_scores, orders.c.strategy_score_id == strategy_scores.c.id)
            .where(orders.c.intent == "open")
            .where(orders.c.status.in_(active))
        ).fetchall()
    heat = 0.0
    for row in rows:
        if row.id in closed_ids:
            continue
        suggestion = row.suggestion_json
        if isinstance(suggestion, str):  # defensive: JSON column round-trip
            suggestion = json.loads(suggestion)
        max_loss = (suggestion or {}).get("max_loss")
        if max_loss:
            heat += float(max_loss) * int(row.quantity)
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
        quantity = 1  # minimum viable lot — it already fits both caps
        floored = ", min-1"
    note = (
        f"sized {quantity}x — base ${base:,.0f}, edge ×{tilt:.1f}, "
        f"dd ×{governor:.1f}{floored}; heat ${open_heat + max_loss_unit * quantity:,.0f}"
        f"/${equity * heat_cap_pct:,.0f}"
    )
    return SizeDecision(quantity, note)
