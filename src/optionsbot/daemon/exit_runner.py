"""Automated exit tick (IBK-129): bot positions → closing orders.

Runs as a sibling of the scan/manage ticks (15-minute cadence — right for
50%-profit / DTE rules; the 1-minute order watcher manages the resulting
working orders). Only ledger-attributed positions are ever touched: a
position is a FILLED open-intent order with no filled close. Manual
positions remain alert-only.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, select

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.execution.engine import combo_mid
from optionsbot.execution.exits import evaluate_exit
from optionsbot.execution.gate import can_execute
from optionsbot.execution.orders import (
    FAILED_TERMINAL_STATUSES,
    OrderRecord,
    get_order,
    net_premium,
    open_close_for,
    stage_close_order,
    transition,
)
from optionsbot.execution.state import load_state
from optionsbot.execution.walk import (
    combo_bid_ask,
    price_increment_for,
    round_to_increment,
    slippage_budget,
)
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.types import OptionQuote
from optionsbot.storage.schema import fills, orders

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExitsTickSummary:
    positions: int
    closes_submitted: int
    errors: int


def _exec_md(context: DaemonContext) -> MarketDataClient | None:
    """Quote source for exit pricing: the EXEC connection (lock-free; the
    scan tick has just released ibkr_lock but /scan can grab it any time).
    Patchable seam for tests."""
    if context.exec_ibkr is None:
        return None
    return MarketDataClient(context.exec_ibkr, context.resolver)


def _open_entries(context: DaemonContext) -> list[OrderRecord]:
    engine = context.engine
    with engine.connect() as conn:
        entry_rows = conn.execute(
            select(orders.c.id)
            .where(orders.c.intent == "open")
            .where(orders.c.status == "filled")
        ).fetchall()
        closed_ids = {
            row.closes_order_id
            for row in conn.execute(
                select(orders.c.closes_order_id)
                .where(orders.c.intent == "close")
                .where(orders.c.status == "filled")
            ).fetchall()
        }
    records = []
    for row in entry_rows:
        if row.id in closed_ids:
            continue
        record = get_order(engine, row.id)
        if record is not None:
            records.append(record)
    return records


def _half_closed(engine: Engine, entry_id: int) -> bool:
    """True when a dead (non-filled terminal) close for this entry recorded
    any fills — the position is partially flat and must not be re-closed at
    full quantity (Opus IBK-129 critical)."""
    with engine.connect() as conn:
        closes = conn.execute(
            select(orders.c.id, orders.c.status)
            .where(orders.c.closes_order_id == entry_id)
        ).fetchall()
        for row in closes:
            if row.status not in FAILED_TERMINAL_STATUSES:
                continue
            has_fill = conn.execute(
                select(fills.c.id).where(fills.c.order_id == row.id).limit(1)
            ).first()
            if has_fill is not None:
                return True
    return False


def _min_dte(legs: list[dict[str, object]], now: datetime) -> int | None:
    dtes = []
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT" or not leg.get("expiry"):
            continue
        expiry = str(leg["expiry"])
        try:
            exp_date = datetime(
                int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]), tzinfo=UTC
            )
        except ValueError:
            continue
        dtes.append((exp_date.date() - now.date()).days)
    return min(dtes) if dtes else None


async def run_exits_tick(context: DaemonContext) -> ExitsTickSummary:
    if context.order_client is None:
        return ExitsTickSummary(0, 0, 0)
    now = datetime.now(UTC)
    verdict = can_execute(context.settings, load_state(context.engine))
    if not verdict.allowed:
        log.debug("exits tick skipped: %s", verdict.reason)
        return ExitsTickSummary(0, 0, 0)
    if not is_market_open(now):
        return ExitsTickSummary(0, 0, 0)
    md = _exec_md(context)
    if md is None:
        return ExitsTickSummary(0, 0, 0)

    entries = _open_entries(context)
    submitted = errors = 0
    for entry in entries:
        try:
            submitted += await _manage_entry(context, md, entry, now)
        except Exception:  # noqa: BLE001 -- one bad position must not starve the rest
            errors += 1
            log.exception("exit evaluation failed for entry #%s", entry.id)
    if entries:
        log.info(
            "exits tick: positions=%d closes=%d errors=%d",
            len(entries), submitted, errors,
        )
    return ExitsTickSummary(positions=len(entries), closes_submitted=submitted, errors=errors)


async def _manage_entry(
    context: DaemonContext, md: MarketDataClient, entry: OrderRecord, now: datetime
) -> int:
    """Evaluate one position; returns 1 if a closing order was submitted."""
    engine = context.engine
    order_client = context.order_client
    if order_client is None:
        return 0
    if open_close_for(engine, entry.id) is not None:
        return 0  # a close is already working — never double-exit
    if _half_closed(engine, entry.id):
        # A close partially filled and then died (abandoned/cancelled): the
        # broker position is HALF flat. Re-staging a full-quantity close
        # would over-close into wrong-way exposure — hand off to the human.
        if entry.id not in context.exit_handoff_warned:
            context.exit_handoff_warned.add(entry.id)
            await _send(
                context,
                f"⚠ position #{entry.id} {entry.symbol} {entry.strategy} is "
                "HALF-CLOSED (a closing order partially filled, then died). "
                "Auto-exit is halted for this position — flatten the remainder "
                "manually in TWS.",
            )
        return 0
    dte = _min_dte(entry.legs, now)
    if dte is None:
        return 0

    total_premium = net_premium(engine, entry.id)
    if total_premium is None or entry.quantity < 1:
        return 0
    entry_net = total_premium / (100 * entry.quantity)  # per-unit, signed

    option_legs = [leg for leg in entry.legs if leg.get("sec_type", "OPT") == "OPT"]
    quotes: dict[tuple[str, float, str], OptionQuote] = {}
    current_net: float | None = None
    try:
        for leg in option_legs:
            spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
            quotes[spec] = await md.get_option_snapshot(
                entry.symbol, spec[0], spec[1], spec[2]  # type: ignore[arg-type]
            )
        current_net = combo_mid(entry.legs, quotes)
    except Exception:  # noqa: BLE001 -- expiry guard must still work quote-blind
        log.warning("exit quotes failed for entry #%s — DTE rules only", entry.id)

    # Quote-freshness gate (IBK-PHASE0-C1). If ANY quote feeding current_net is
    # older than the threshold, drop current_net to None: evaluate_exit then
    # takes its quote-blind path (expiry/DTE only), so a TP/soft stop can never
    # be priced off a stale mid. Always log the timestamps the decision used.
    max_age = context.settings.execution.exit_quote_max_age_seconds
    quote_ages = {spec: (now - q.ts).total_seconds() for spec, q in quotes.items()}
    if quotes:
        log.info(
            "exit quotes for entry #%s: %s",
            entry.id,
            ", ".join(
                f"{spec[1]}{spec[2]}@{q.ts.isoformat()} (age {quote_ages[spec]:.0f}s)"
                for spec, q in quotes.items()
            ),
        )
    stale = bool(
        max_age > 0
        and current_net is not None
        and any(age > max_age for age in quote_ages.values())
    )
    if stale:
        oldest = max(quote_ages.values())
        if entry.id not in context.exit_stale_warned:
            context.exit_stale_warned.add(entry.id)
            await _send(
                context,
                f"⚠ exit pricing for #{entry.id} {entry.symbol} {entry.strategy} "
                f"suppressed: quotes are STALE (oldest {oldest:.0f}s > "
                f"{max_age}s). TP/stop deferred; expiry/DTE rules still active.",
            )
        log.warning(
            "exit #%s quote-priced exit suppressed (stale: oldest %.0fs > %ss)",
            entry.id, oldest, max_age,
        )
        current_net = None
        quotes = {}  # stale quotes must not price the close/walk either
    else:
        context.exit_stale_warned.discard(entry.id)

    reason = evaluate_exit(
        entry_net=entry_net, current_net=current_net, dte=dte,
        settings=context.settings,
    )
    if reason is None:
        return 0

    # Price the FLIPPED structure. Closing a credit pays; closing a debit
    # collects. With no usable quote (expiry guard path) fall back to the
    # entry net — the walk re-anchors from live quotes immediately anyway.
    # Rounded to the symbol tick: IBKR rejects sub-increment limits (Error 110).
    increment = price_increment_for(entry.symbol)
    close_net = round_to_increment(
        -(current_net if current_net is not None else entry_net), increment
    )
    limit_price = -close_net  # BAG convention: limit = -net

    close = stage_close_order(engine, entry, now=now)
    transition(engine, close.id, "submitting", now=now)
    try:
        placed = await order_client.place_combo_limit(
            entry.symbol,
            close.legs,
            quantity=close.quantity,
            limit_price=limit_price,
            order_ref=close.order_ref or f"obot-{close.id}",
        )
    except Exception as exc:  # noqa: BLE001 -- a failed close lands in the ledger + retries next tick
        transition(engine, close.id, "skipped", error=str(exc), now=now)
        await _send(context, f"⚠ closing #{entry.id} failed to place: {exc} — will retry")
        return 0
    transition(engine, close.id, "submitted", ib_order_id=placed.ib_order_id, now=now)

    # Walk the close like an entry (IBK-127).
    cfg = context.settings.execution
    if cfg.walk_max_steps > 0:
        flipped_nbbo = combo_bid_ask(close.legs, quotes) if quotes else None
        budget = (
            slippage_budget(
                flipped_nbbo[0], flipped_nbbo[1],
                frac=cfg.max_slippage_spread_frac,
                abs_cap=cfg.max_slippage_abs, increment=increment,
            )
            if flipped_nbbo is not None
            else increment
        )
        from optionsbot.execution.walk import run_price_walk

        task = asyncio.create_task(
            run_price_walk(
                engine=engine, settings=context.settings,
                order_client=order_client, md=md, symbol=entry.symbol,
                legs=close.legs, order_id=close.id,
                ib_order_id=placed.ib_order_id, decision_mid=close_net,
                budget=budget, increment=increment,
            )
        )
        context.walk_tasks.add(task)
        task.add_done_callback(context.walk_tasks.discard)

    await _send(
        context,
        f"📤 closing #{entry.id} {entry.symbol} {entry.strategy} "
        f"{entry.quantity}x — {reason} (close order #{close.id})",
    )
    return 1


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001
        log.exception("exit notification failed")
