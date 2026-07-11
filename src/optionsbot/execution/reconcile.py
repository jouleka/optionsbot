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

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, select

from optionsbot.execution.orders import (
    FAILED_TERMINAL_STATUSES,
    IllegalOrderTransition,
    get_order,
    open_position_exposure,
    record_fill,
    set_fill_commission,
    transition,
)
from optionsbot.execution.state import trip_kill
from optionsbot.execution.tracker import map_ib_status, row_id_from_ref
from optionsbot.execution.walk import resume_walks
from optionsbot.storage.schema import fills, orders

if TYPE_CHECKING:
    from optionsbot.ibkr.orders import OrderClient
    from optionsbot.ibkr.types import PortfolioPosition

log = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None]]

_STALE_STAGED_AFTER = timedelta(minutes=30)

# An /execute may be suspended between transition(submitting) and the broker
# ack when this pass snapshots open orders (the place path awaits connect/
# qualify/rate-limit). A row younger than this grace is treated as in-flight
# and left for the next pass — resolving it as failed would strand a REAL
# order at the broker under a terminal ledger row (Opus C1).
_INFLIGHT_GRACE = timedelta(seconds=60)


@dataclass(frozen=True, slots=True)
class ReconcileSummary:
    adopted: int
    foreign: int
    fills_replayed: int
    resolved: int  # ledger rows whose status was corrected
    mismatches: int
    orphan_positions: int = 0  # broker positions with no open ledger row


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
    """Prove exact per-contract, side, and ratio completion for every option leg."""
    if isinstance(quantity, bool) or quantity <= 0:
        return False
    expected: dict[tuple[int, str], int] = {}
    seen_contracts: set[int] = set()
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        try:
            raw_con_id = leg["con_id"]
            raw_ratio = leg.get("quantity", 1)
            con_id = int(raw_con_id)
            ratio = int(raw_ratio)
            side = str(leg["side"]).upper()
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            isinstance(raw_con_id, bool)
            or con_id <= 0
            or con_id != raw_con_id
            or con_id in seen_contracts
            or isinstance(raw_ratio, bool)
            or ratio <= 0
            or ratio != raw_ratio
            or side not in {"BUY", "SELL"}
        ):
            return False
        seen_contracts.add(con_id)
        expected[(con_id, side)] = ratio * quantity
    if not expected:
        return False

    with engine.connect() as conn:
        rows = conn.execute(
            select(fills.c.leg_con_id, fills.c.side, fills.c.qty).where(
                fills.c.order_id == order_id
            )
        ).fetchall()
    actual: dict[tuple[int, str], int] = {}
    for row in rows:
        try:
            raw_con_id = row.leg_con_id
            raw_qty = row.qty
            con_id = int(raw_con_id)
            fill_qty = int(raw_qty)
            side = str(row.side).upper()
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            isinstance(raw_con_id, bool)
            or con_id <= 0
            or con_id != raw_con_id
            or isinstance(raw_qty, bool)
            or fill_qty <= 0
            or fill_qty != raw_qty
            or side not in {"BUY", "SELL"}
        ):
            return False
        key = (con_id, side)
        actual[key] = actual.get(key, 0) + fill_qty
    return actual == expected


async def _reconcile_once(
    engine: Engine,
    order_client: OrderClient,
    *,
    notify: Notify | None = None,
    now: datetime | None = None,
    walk_md: Any = None,
    walk_tasks: Any = None,
    walk_resume: Callable[..., Awaitable[int]] | None = None,
    settings: Any = None,
    positions_snapshot: Callable[[], Awaitable[list[PortfolioPosition]]] | None = None,
) -> ReconcileSummary:
    """One full broker↔ledger convergence pass. Never raises."""
    ts_now = now if now is not None else datetime.now(UTC)
    adopted = foreign = replayed = resolved = mismatches = 0
    try:
        broker_orders = await order_client.adopt_open_orders()
    except Exception as exc:  # noqa: BLE001 -- unavailable broker state fails closed
        log.exception("reconcile: adopt_open_orders failed")
        reason = f"reconcile open-order snapshot unavailable: {exc}"
        trip_kill(engine, reason, now=ts_now)
        await _send(notify, f"🛑 KILL SWITCH: {reason}")
        return ReconcileSummary(0, 0, 0, 0, 1)
    if not isinstance(broker_orders, (list, tuple)):
        reason = "reconcile open-order snapshot malformed"
        trip_kill(engine, reason, now=ts_now)
        await _send(notify, f"🛑 KILL SWITCH: {reason}")
        return ReconcileSummary(0, 0, 0, 0, 1)

    at_broker: dict[int, tuple[int, str]] = {}  # ledger row id -> (ib id, ib status)
    for broker_row in broker_orders:
        if not isinstance(broker_row, (list, tuple)) or len(broker_row) != 3:
            reason = "reconcile open-order snapshot contains a malformed row"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            return ReconcileSummary(adopted, foreign, 0, 0, 1)
        ib_order_id, ref, ib_status = broker_row
        if (
            type(ib_order_id) is not int
            or ib_order_id < 0
            or (ref is not None and not isinstance(ref, str))
            or not isinstance(ib_status, str)
            or not ib_status.strip()
        ):
            reason = "reconcile open-order snapshot contains invalid identity fields"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            return ReconcileSummary(adopted, foreign, 0, 0, 1)
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
        if row_id in at_broker:
            mismatches += 1
            reason = (
                f"reconcile duplicate broker orders claim ledger order #{row_id}: "
                f"{at_broker[row_id][0]} and {ib_order_id}"
            )
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            continue
        at_broker[row_id] = (ib_order_id, ib_status)

    # Work-stream D1: re-attach any persisted price-walks for orders we just
    # confirmed are still at the broker. The walk's in-memory asyncio task died
    # with the previous process; resume_walks rebuilds it (resuming from the
    # persisted step) so the order isn't orphaned at the decision mid until TTL.
    if walk_md is not None and walk_tasks is not None:
        resume = walk_resume if walk_resume is not None else resume_walks
        try:
            await resume(
                engine=engine, settings=settings, order_client=order_client,
                md=walk_md, walk_tasks=walk_tasks, notify=notify,
            )
        except Exception as exc:  # noqa: BLE001 -- unmanaged working order halts
            log.exception("reconcile: walk resume failed")
            mismatches += 1
            reason = f"reconcile persisted price-walk resume failed: {exc}"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")

    # Sync ledger rows that ARE at the broker (e.g. submitting -> submitted).
    for row_id, (ib_order_id, ib_status) in at_broker.items():
        record = get_order(engine, row_id)
        if record is None:
            mismatches += 1
            reason = f"reconcile broker order obot-{row_id} has no ledger row"
            trip_kill(engine, reason, now=ts_now)
            await _send(
                notify,
                f"🛑 KILL SWITCH: broker order obot-{row_id} has no ledger row — "
                "manual reconciliation required",
            )
            continue
        target = map_ib_status(ib_status, 0, 1)  # working ack mapping
        if target is None:
            mismatches += 1
            reason = (
                f"reconcile broker order obot-{row_id} has unknown status {ib_status!r}"
            )
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            continue
        if record.status == target:
            continue
        if record.status == "partial" and target == "submitted":
            # We have no fill counts here; a partially-filled at-broker order
            # is already correctly "working" — nothing to sync.
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
    except Exception as exc:  # noqa: BLE001 -- unavailable broker state fails closed
        log.exception("reconcile: recent_executions failed")
        reason = f"reconcile execution snapshot unavailable: {exc}"
        trip_kill(engine, reason, now=ts_now)
        await _send(notify, f"🛑 KILL SWITCH: {reason}")
        return ReconcileSummary(adopted, foreign, replayed, resolved, mismatches + 1)
    if not isinstance(executions, (list, tuple)):
        reason = "reconcile execution snapshot malformed"
        trip_kill(engine, reason, now=ts_now)
        await _send(notify, f"🛑 KILL SWITCH: {reason}")
        return ReconcileSummary(adopted, foreign, replayed, resolved, mismatches + 1)
    for execution in executions:
        try:
            sec_type = execution.sec_type
            order_ref = execution.order_ref
            exec_id = execution.exec_id
            side = execution.side
            price = execution.price
            raw_qty = execution.qty
            execution_ts = execution.ts
            raw_con_id = execution.con_id
            commission = execution.commission
        except AttributeError:
            reason = "reconcile execution snapshot contains a malformed row"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            return ReconcileSummary(adopted, foreign, replayed, resolved, mismatches + 1)
        if (
            not isinstance(sec_type, str)
            or sec_type not in {"OPT", "BAG"}
            or not isinstance(order_ref, str)
            or not isinstance(exec_id, str)
            or not exec_id.strip()
            or side not in {"BUY", "SELL"}
            or not isinstance(price, (int, float))
            or isinstance(price, bool)
            or not math.isfinite(float(price))
            or type(raw_qty) is not int
            or raw_qty <= 0
            or not isinstance(execution_ts, datetime)
            or type(raw_con_id) is not int
            or raw_con_id < (0 if sec_type == "BAG" else 1)
            or (
                commission is not None
                and (
                    not isinstance(commission, (int, float))
                    or isinstance(commission, bool)
                    or not math.isfinite(float(commission))
                    or commission < 0
                )
            )
        ):
            reason = "reconcile execution snapshot contains invalid evidence"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            return ReconcileSummary(adopted, foreign, replayed, resolved, mismatches + 1)
        if sec_type == "BAG":
            continue
        row_id = row_id_from_ref(order_ref)
        if row_id is None:
            continue
        record = get_order(engine, row_id)
        if record is None:
            mismatches += 1
            reason = f"reconcile execution {exec_id!r} references missing order #{row_id}"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
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
        if record.status in FAILED_TERMINAL_STATUSES:
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
                   orders.c.legs_json, orders.c.staged_ts, orders.c.submitted_ts)
            .where(orders.c.status.in_(sorted(set(map(str, (
                "staged", "submitting", "submitted", "partial"))) )))
        ).fetchall()
    for row in rows:
        if row.id in at_broker:
            continue
        legs = list(row.legs_json or [])
        anchor = row.submitted_ts or row.staged_ts
        if anchor is not None and anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=UTC)
        in_flight = anchor is None or (ts_now - anchor) < _INFLIGHT_GRACE
        try:
            if row.status in ("submitted", "partial"):
                if _fills_complete(engine, row.id, row.quantity, legs):
                    transition(engine, row.id, "filled", now=ts_now)
                    resolved += 1
                elif in_flight:
                    continue  # an /execute may still be mid-place — next pass
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
                    resolved += 1
                elif in_flight:
                    continue
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

    # Work-stream D2: position-level compare. Beyond orders, ask the broker for
    # OPEN POSITIONS and confirm every one maps to a leg the ledger believes is
    # open. A broker position with no matching open ledger leg is a forgotten
    # position — the most dangerous mismatch after a fill-the-books-deny, since
    # it is unmanaged by the exit engine. Alert and trip the kill switch.
    orphan_positions = 0
    if positions_snapshot is not None:
        try:
            broker_positions = await positions_snapshot()
        except Exception as exc:  # noqa: BLE001 -- unavailable state fails closed
            log.exception("reconcile: positions snapshot failed")
            reason = f"reconcile position snapshot unavailable: {exc}"
            trip_kill(engine, reason)
            mismatches += 1
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
        else:
            ledger_exposure = open_position_exposure(engine)
            broker_map: dict[
                int, tuple[tuple[str, str, float, str], int]
            ] = {}
            broker_valid = isinstance(broker_positions, (list, tuple))
            for pos in broker_positions if broker_valid else ():
                try:
                    sec_type = pos.sec_type
                    raw_position = pos.position
                    raw_con_id = pos.con_id
                    expiry = pos.expiry
                    raw_strike = pos.strike
                    right = pos.right
                    symbol = pos.symbol
                except AttributeError:
                    broker_valid = False
                    break
                if sec_type != "OPT":
                    continue
                if (
                    not isinstance(raw_position, (int, float))
                    or isinstance(raw_position, bool)
                    or not math.isfinite(float(raw_position))
                    or not float(raw_position).is_integer()
                ):
                    broker_valid = False
                    break
                if raw_position == 0:
                    continue
                if (
                    type(raw_con_id) is not int
                    or raw_con_id <= 0
                    or not isinstance(expiry, str)
                    or not expiry
                    or not isinstance(raw_strike, (int, float))
                    or isinstance(raw_strike, bool)
                    or not math.isfinite(float(raw_strike))
                    or raw_strike <= 0
                    or right not in {"C", "P"}
                    or not isinstance(symbol, str)
                    or not symbol
                ):
                    broker_valid = False
                    break
                con_id = raw_con_id
                quantity = int(raw_position)
                spec = (symbol, expiry, float(raw_strike), right)
                prior = broker_map.get(con_id)
                if prior is not None and prior[0] != spec:
                    broker_valid = False
                    break
                broker_map[con_id] = (
                    spec,
                    (prior[1] if prior else 0) + quantity,
                )
            broker_exposure = (
                {
                    con_id: value
                    for con_id, value in broker_map.items()
                    if value[1] != 0
                }
                if broker_valid
                else None
            )

            if ledger_exposure is None or broker_exposure is None:
                reason = (
                    "reconcile position attribution incomplete; exact broker/ledger "
                    "agreement cannot be proven"
                )
                trip_kill(engine, reason)
                mismatches += 1
                await _send(notify, f"🛑 KILL SWITCH: {reason}")
            elif ledger_exposure != broker_exposure:
                differing = set(ledger_exposure) | set(broker_exposure)
                differing = {
                    con_id
                    for con_id in differing
                    if ledger_exposure.get(con_id) != broker_exposure.get(con_id)
                }
                orphan_positions = len(differing)
                mismatches += 1
                reason = (
                    "reconcile exact position mismatch for contract IDs "
                    f"{sorted(differing)}; broker={broker_exposure!r}, "
                    f"ledger={ledger_exposure!r}"
                )
                trip_kill(engine, reason)
                await _send(
                    notify,
                    "🛑 KILL SWITCH: exact broker/ledger option exposure mismatch — "
                    "inspect /positions and ledger, reconcile manually, then /arm.",
                )

    if adopted or foreign or replayed or resolved or mismatches or orphan_positions:
        log.info(
            "reconcile: adopted=%d foreign=%d fills=%d resolved=%d mismatches=%d "
            "orphan_positions=%d",
            adopted, foreign, replayed, resolved, mismatches, orphan_positions,
        )
    return ReconcileSummary(
        adopted=adopted, foreign=foreign, fills_replayed=replayed,
        resolved=resolved, mismatches=mismatches, orphan_positions=orphan_positions,
    )


async def reconcile(
    engine: Engine,
    order_client: OrderClient,
    *,
    notify: Notify | None = None,
    now: datetime | None = None,
    walk_md: Any = None,
    walk_tasks: Any = None,
    walk_resume: Callable[..., Awaitable[int]] | None = None,
    settings: Any = None,
    positions_snapshot: Callable[[], Awaitable[list[PortfolioPosition]]] | None = None,
) -> ReconcileSummary:
    """Run reconciliation and convert every unexpected defect into a halt."""
    try:
        return await _reconcile_once(
            engine,
            order_client,
            notify=notify,
            now=now,
            walk_md=walk_md,
            walk_tasks=walk_tasks,
            walk_resume=walk_resume,
            settings=settings,
            positions_snapshot=positions_snapshot,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- public boundary must fail closed
        log.exception("reconcile: unexpected processing failure")
        reason = f"reconcile processing unavailable: {exc}"
        try:
            trip_kill(engine, reason, now=now)
        except Exception:  # noqa: BLE001 -- still alert and report mismatch
            log.exception("reconcile: failed to persist kill switch")
        await _send(notify, f"🛑 KILL SWITCH: {reason}")
        return ReconcileSummary(0, 0, 0, 0, 1)
