"""Broker reconciliation (IBK-128): the ledger converges to the broker.

Runs at daemon startup and periodically from the order watcher. Adopts
working orders across restarts (re-arming modify/cancel), resolves
crash-orphaned ``submitting``/``staged`` rows (never blind-resubmits),
replays missed executions idempotently, warns about foreign orders (never
auto-cancels someone's manual trade), and trips the kill switch on the one
truly dangerous mismatch: a real fill for an order the ledger recorded as
failed — a position the bot's books deny.

NOT re-exported from ``optionsbot.execution.__init__`` (imports ibkr
types; same import-graph reasoning as tracker.py/engine.py/walk.py).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, select

from optionsbot.execution.orders import (
    IllegalOrderTransition,
    get_order,
    record_fill,
    set_fill_commission,
    transition,
)
from optionsbot.execution.state import trip_kill
from optionsbot.execution.tracker import map_ib_status, row_id_from_ref
from optionsbot.storage.schema import fills, orders

if TYPE_CHECKING:
    from optionsbot.ibkr.orders import OrderClient

log = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None]]

# Failed terminals: a NEW fill arriving for one of these means the broker
# holds a position the ledger denies — the dangerous mismatch.
_FAILED_TERMINALS = frozenset({"skipped", "rejected", "cancelled", "abandoned"})

_STALE_STAGED_AFTER = timedelta(minutes=30)


@dataclass(frozen=True, slots=True)
class ReconcileSummary:
    adopted: int
    foreign: int
    fills_replayed: int
    resolved: int  # ledger rows whose status was corrected
    mismatches: int


async def _send(notify: Notify | None, text: str) -> None:
    if notify is None:
        return
    try:
        await notify(text)
    except Exception:  # noqa: BLE001 -- notification failure must not stop reconcile
        log.exception("reconcile notification failed")


def _fills_complete(
    engine: Engine, order_id: int, quantity: int, legs: list[dict[str, Any]]
) -> bool:
    n_option_legs = sum(1 for leg in legs if leg.get("sec_type", "OPT") == "OPT")
    if n_option_legs == 0:
        return False
    with engine.connect() as conn:
        total = conn.execute(
            select(fills.c.qty).where(fills.c.order_id == order_id)
        ).fetchall()
    filled_qty = sum(int(r.qty) for r in total)
    return filled_qty >= quantity * n_option_legs


async def reconcile(
    engine: Engine,
    order_client: OrderClient,
    *,
    notify: Notify | None = None,
    now: datetime | None = None,
) -> ReconcileSummary:
    """One full broker↔ledger convergence pass. Never raises."""
    ts_now = now if now is not None else datetime.now(UTC)
    adopted = foreign = replayed = resolved = mismatches = 0
    try:
        broker_orders = await order_client.adopt_open_orders()
    except Exception:  # noqa: BLE001 -- a dead gateway must not kill the caller
        log.exception("reconcile: adopt_open_orders failed")
        return ReconcileSummary(0, 0, 0, 0, 0)

    at_broker: dict[int, tuple[int, str]] = {}  # ledger row id -> (ib id, ib status)
    for ib_order_id, ref, ib_status in broker_orders:
        row_id = row_id_from_ref(ref)
        if row_id is None:
            foreign += 1
            await _send(
                notify,
                f"⚠ open order at IBKR not placed by the bot "
                f"(id {ib_order_id}, ref {ref or '—'}) — leaving it alone",
            )
            continue
        adopted += 1
        at_broker[row_id] = (ib_order_id, ib_status)

    # Sync ledger rows that ARE at the broker (e.g. submitting -> submitted).
    for row_id, (ib_order_id, ib_status) in at_broker.items():
        record = get_order(engine, row_id)
        if record is None:
            await _send(
                notify,
                f"⚠ broker order obot-{row_id} has no ledger row — manual check needed",
            )
            continue
        target = map_ib_status(ib_status, 0, 1)  # working ack mapping
        if target is None or record.status == target:
            continue
        try:
            transition(engine, row_id, target, ib_order_id=ib_order_id, now=ts_now)
            resolved += 1
        except IllegalOrderTransition:
            log.warning(
                "reconcile: cannot move order %s %s -> %s",
                row_id, record.status, target,
            )

    # Replay today's executions (execId dedupe = idempotent).
    try:
        executions = await order_client.recent_executions()
    except Exception:  # noqa: BLE001
        log.exception("reconcile: recent_executions failed")
        executions = []
    for execution in executions:
        if execution.sec_type == "BAG":
            continue
        row_id = row_id_from_ref(execution.order_ref)
        if row_id is None:
            continue
        record = get_order(engine, row_id)
        if record is None:
            continue
        was_new = record_fill(
            engine, row_id, exec_id=execution.exec_id, side=execution.side,
            price=execution.price, qty=execution.qty, ts=execution.ts,
            leg_con_id=execution.con_id,
        )
        if execution.commission is not None:
            set_fill_commission(engine, execution.exec_id, execution.commission)
        if not was_new:
            continue
        replayed += 1
        if record.status in _FAILED_TERMINALS:
            # The broker filled an order the ledger recorded as failed: a
            # real position the books deny. Stop everything, human required.
            mismatches += 1
            trip_kill(
                engine,
                f"reconcile mismatch: fill {execution.exec_id} for order "
                f"#{row_id} in status {record.status}",
            )
            await _send(
                notify,
                f"🛑 KILL SWITCH: broker reported a fill for order #{row_id} "
                f"which the ledger has as {record.status} — positions and "
                "ledger disagree. /positions to inspect; /arm after resolving.",
            )

    # Resolve ledger rows the broker no longer has.
    with engine.connect() as conn:
        rows = conn.execute(
            select(orders.c.id, orders.c.status, orders.c.quantity,
                   orders.c.legs_json, orders.c.staged_ts)
            .where(orders.c.status.in_(sorted(set(map(str, (
                "staged", "submitting", "submitted", "partial"))) )))
        ).fetchall()
    for row in rows:
        if row.id in at_broker:
            continue
        legs = list(row.legs_json or [])
        try:
            if row.status in ("submitted", "partial"):
                if _fills_complete(engine, row.id, row.quantity, legs):
                    transition(engine, row.id, "filled", now=ts_now)
                else:
                    transition(
                        engine, row.id, "cancelled",
                        error="not at broker after restart (reconciled)",
                        now=ts_now,
                    )
                resolved += 1
            elif row.status == "submitting":
                if _fills_complete(engine, row.id, row.quantity, legs):
                    transition(engine, row.id, "submitted", now=ts_now)
                    transition(engine, row.id, "filled", now=ts_now)
                else:
                    transition(
                        engine, row.id, "skipped",
                        error="no broker record after crash — never resubmitted (reconciled)",
                        now=ts_now,
                    )
                resolved += 1
            elif row.status == "staged":
                staged_ts = row.staged_ts
                if staged_ts is not None and staged_ts.tzinfo is None:
                    staged_ts = staged_ts.replace(tzinfo=UTC)
                if staged_ts is not None and ts_now - staged_ts > _STALE_STAGED_AFTER:
                    transition(
                        engine, row.id, "skipped",
                        error="stale staged row (crashed before submit) — reconciled",
                        now=ts_now,
                    )
                    resolved += 1
        except IllegalOrderTransition:
            log.warning("reconcile: race resolving order %s (%s)", row.id, row.status)

    if adopted or foreign or replayed or resolved or mismatches:
        log.info(
            "reconcile: adopted=%d foreign=%d fills=%d resolved=%d mismatches=%d",
            adopted, foreign, replayed, resolved, mismatches,
        )
    return ReconcileSummary(
        adopted=adopted, foreign=foreign, fills_replayed=replayed,
        resolved=resolved, mismatches=mismatches,
    )
