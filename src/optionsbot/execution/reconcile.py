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
from optionsbot.execution.state import clear_kill, load_state, trip_kill
from optionsbot.execution.tracker import map_ib_status, row_id_from_ref
from optionsbot.execution.walk import resume_walks
from optionsbot.ibkr.types import OpenOrderSnapshot, ledger_row_id_from_ref
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
_TRANSIENT_GATEWAY_DISCONNECT_KILL = (
    "reconcile open-order snapshot unavailable: Socket disconnect"
)
_KNOWN_BROKER_SECURITY_TYPES = frozenset(
    {"OPT", "STK", "FUT", "FOP", "CASH", "CFD", "BOND", "CMDTY", "FUND", "BAG"}
)


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


def _persisted_order_semantics_valid(record: Any) -> bool:
    if type(record.quantity) is not int or record.quantity <= 0:
        return False
    legs = record.legs
    if not isinstance(legs, list) or not legs:
        return False
    seen_con_ids: set[int] = set()
    option_count = 0
    for leg in legs:
        if not isinstance(leg, dict):
            return False
        sec_type = leg.get("sec_type", "OPT")
        if sec_type == "STK":
            if leg.get("symbol") != record.symbol:
                return False
            continue
        if sec_type != "OPT":
            return False
        option_count += 1
        con_id = leg.get("con_id")
        strike = leg.get("strike")
        ratio = leg.get("quantity")
        if (
            leg.get("symbol") != record.symbol
            or leg.get("side") not in {"buy", "sell"}
            or not isinstance(leg.get("expiry"), str)
            or len(leg["expiry"]) != 8
            or not leg["expiry"].isdigit()
            or not isinstance(strike, (int, float))
            or isinstance(strike, bool)
            or not math.isfinite(float(strike))
            or strike <= 0
            or leg.get("right") not in {"C", "P"}
            or type(ratio) is not int
            or ratio <= 0
            or type(con_id) is not int
            or con_id <= 0
            or con_id in seen_con_ids
            or leg.get("multiplier") != 100
            or leg.get("currency") != "USD"
        ):
            return False
        seen_con_ids.add(con_id)
    return option_count > 0


def _broker_order_matches_ledger(snapshot: OpenOrderSnapshot, record: Any) -> bool:
    """Require the exact broker contract/order terms persisted for this intent."""
    if not _persisted_order_semantics_valid(record):
        return False
    raw_limit = record.limit_price
    if (
        not isinstance(raw_limit, (int, float))
        or isinstance(raw_limit, bool)
        or not math.isfinite(float(raw_limit))
    ):
        return False
    option_legs = [leg for leg in record.legs if leg.get("sec_type", "OPT") == "OPT"]
    common = (
        snapshot.symbol == record.symbol
        and snapshot.currency == "USD"
        and snapshot.exchange == "SMART"
        and snapshot.order_type == "LMT"
        and snapshot.tif == "DAY"
    )
    if not common:
        return False
    if len(option_legs) == 1:
        leg = option_legs[0]
        return (
            snapshot.sec_type == "OPT"
            and snapshot.contract_con_id == leg["con_id"]
            and snapshot.multiplier == leg["multiplier"]
            and snapshot.expiry == leg["expiry"]
            and snapshot.strike == float(leg["strike"])
            and snapshot.right == leg["right"]
            and snapshot.combo_legs == ()
            and snapshot.order_action == str(leg["side"]).upper()
            and snapshot.total_quantity == record.quantity * leg["quantity"]
            and snapshot.limit_price is not None
            and math.isclose(
                snapshot.limit_price,
                abs(float(raw_limit)),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    expected_combo = tuple(
        (leg["con_id"], leg["quantity"], str(leg["side"]).upper(), "SMART")
        for leg in option_legs
    )
    return (
        snapshot.sec_type == "BAG"
        and snapshot.contract_con_id is None
        and snapshot.multiplier is None
        and snapshot.expiry is None
        and snapshot.strike is None
        and snapshot.right is None
        and snapshot.combo_legs == expected_combo
        and snapshot.order_action == "BUY"
        and snapshot.total_quantity == record.quantity
        and snapshot.limit_price is not None
        and math.isclose(
            snapshot.limit_price,
            float(raw_limit),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )


def _open_order_snapshot_valid(snapshot: OpenOrderSnapshot) -> bool:
    """Validate exact runtime types at the reconciliation trust boundary."""
    if (
        type(snapshot.ib_order_id) is not int
        or snapshot.ib_order_id < 0
        or (snapshot.order_ref is not None and not isinstance(snapshot.order_ref, str))
        or not isinstance(snapshot.status, str)
        or not snapshot.status.strip()
    ):
        return False
    row_id = ledger_row_id_from_ref(snapshot.order_ref)
    if row_id is None:
        return not (
            isinstance(snapshot.order_ref, str)
            and snapshot.order_ref.startswith("obot-")
        )
    if (
        snapshot.ib_order_id <= 0
        or
        snapshot.order_ref != f"obot-{row_id}"
        or snapshot.sec_type not in {"OPT", "BAG"}
        or not isinstance(snapshot.symbol, str)
        or not snapshot.symbol.strip()
        or snapshot.currency != "USD"
        or snapshot.exchange != "SMART"
        or snapshot.order_action not in {"BUY", "SELL"}
        or type(snapshot.total_quantity) is not int
        or snapshot.total_quantity <= 0
        or snapshot.order_type != "LMT"
        or snapshot.tif != "DAY"
        or not isinstance(snapshot.limit_price, (int, float))
        or isinstance(snapshot.limit_price, bool)
        or not math.isfinite(float(snapshot.limit_price))
    ):
        return False
    if snapshot.sec_type == "BAG":
        if (
            snapshot.contract_con_id is not None
            or snapshot.multiplier is not None
            or snapshot.expiry is not None
            or snapshot.strike is not None
            or snapshot.right is not None
            or not isinstance(snapshot.combo_legs, tuple)
            or len(snapshot.combo_legs) < 2
        ):
            return False
        seen: set[int] = set()
        for leg in snapshot.combo_legs:
            if (
                not isinstance(leg, tuple)
                or len(leg) != 4
                or type(leg[0]) is not int
                or leg[0] <= 0
                or leg[0] in seen
                or type(leg[1]) is not int
                or leg[1] <= 0
                or leg[2] not in {"BUY", "SELL"}
                or leg[3] != "SMART"
            ):
                return False
            seen.add(leg[0])
        return True
    return (
        type(snapshot.contract_con_id) is int
        and snapshot.contract_con_id > 0
        and type(snapshot.multiplier) is int
        and snapshot.multiplier > 0
        and isinstance(snapshot.expiry, str)
        and len(snapshot.expiry) == 8
        and snapshot.expiry.isascii()
        and snapshot.expiry.isdecimal()
        and isinstance(snapshot.strike, (int, float))
        and not isinstance(snapshot.strike, bool)
        and math.isfinite(float(snapshot.strike))
        and snapshot.strike > 0
        and snapshot.right in {"C", "P"}
        and snapshot.combo_legs == ()
    )


def _fills_complete(
    engine: Engine, order_id: int, quantity: int, legs: list[dict[str, Any]]
) -> bool:
    """Prove exact per-contract, side, and ratio completion for every option leg."""
    if type(quantity) is not int or quantity <= 0:
        return False
    expected: dict[tuple[int, str], int] = {}
    seen_contracts: set[int] = set()
    for leg in legs:
        if not isinstance(leg, dict):
            return False
        sec_type = leg.get("sec_type", "OPT")
        if sec_type == "STK":
            continue
        if sec_type != "OPT":
            return False
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

    at_broker: dict[int, OpenOrderSnapshot] = {}
    authorization_candidates: list[OpenOrderSnapshot] = []
    broker_id_owner: dict[int, int] = {}  # ib order id -> ledger row id
    for broker_row in broker_orders:
        if (
            not isinstance(broker_row, OpenOrderSnapshot)
            or not _open_order_snapshot_valid(broker_row)
        ):
            reason = "reconcile open-order snapshot contains a malformed row"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            return ReconcileSummary(adopted, foreign, 0, 0, 1)
        ib_order_id = broker_row.ib_order_id
        ref = broker_row.order_ref
        ib_status = broker_row.status
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
                f"{at_broker[row_id].ib_order_id} and {ib_order_id}"
            )
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            continue
        owner = broker_id_owner.get(ib_order_id)
        if owner is not None and owner != row_id:
            mismatches += 1
            reason = (
                f"reconcile broker order id {ib_order_id} claims multiple ledger rows: "
                f"#{owner} and #{row_id}"
            )
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            continue
        broker_id_owner[ib_order_id] = row_id
        at_broker[row_id] = broker_row

    # Sync ledger rows that ARE at the broker (e.g. submitting -> submitted).
    for row_id, broker_row in at_broker.items():
        ib_order_id = broker_row.ib_order_id
        ib_status = broker_row.status
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
        expected_ref = f"obot-{row_id}"
        if (
            record.order_ref != expected_ref
            or (
                record.ib_order_id is not None
                and record.ib_order_id != ib_order_id
            )
        ):
            mismatches += 1
            reason = (
                f"reconcile broker identity conflicts with ledger order #{row_id}: "
                f"ref={record.order_ref!r}, ledger_ib_id={record.ib_order_id}, "
                f"broker_ib_id={ib_order_id}"
            )
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            continue
        if not _persisted_order_semantics_valid(record):
            mismatches += 1
            reason = f"reconcile ledger order #{row_id} has invalid persisted semantics"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
            continue
        if not _broker_order_matches_ledger(broker_row, record):
            mismatches += 1
            reason = f"reconcile broker terms conflict with ledger order #{row_id}"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
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
            authorization_candidates.append(broker_row)
            continue
        if record.status == "partial" and target == "submitted":
            # We have no fill counts here; a partially-filled at-broker order
            # is already correctly "working" — nothing to sync.
            authorization_candidates.append(broker_row)
            continue
        try:
            transition(engine, row_id, target, ib_order_id=ib_order_id, now=ts_now)
            resolved += 1
            authorization_candidates.append(broker_row)
        except IllegalOrderTransition as exc:
            log.warning(
                "reconcile: cannot move order %s %s -> %s",
                row_id, record.status, target,
            )
            mismatches += 1
            reason = f"reconcile illegal broker/ledger state for order #{row_id}: {exc}"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")

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
        except IllegalOrderTransition as exc:
            log.warning("reconcile: race resolving order %s (%s)", row.id, row.status)
            mismatches += 1
            reason = f"reconcile ledger transition race for order #{row.id}: {exc}"
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")

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
                if (
                    not isinstance(sec_type, str)
                    or sec_type not in _KNOWN_BROKER_SECURITY_TYPES
                ):
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

    # A scheduled Gateway recycle can interrupt the opening snapshot and trip
    # the persistent switch. Re-arm only this exact transient reason, and only
    # after a later pass has proved all broker orders, executions, and positions
    # agree. Every other kill reason remains manual.
    state = load_state(engine)
    if (
        mismatches == 0
        and positions_snapshot is not None
        and state.killed
        and state.reason == _TRANSIENT_GATEWAY_DISCONNECT_KILL
    ):
        state = clear_kill(engine, now=ts_now)
        log.info("reconcile: auto-rearmed after clean post-disconnect snapshot")
        await _send(
            notify,
            "✅ execution re-armed after a complete clean broker reconciliation "
            "following the transient Gateway disconnect",
        )

    # Mutation authority and walk resumption are the final commit point: no
    # broker order becomes mutable until every order, execution, and position
    # check in this reconciliation pass has succeeded.
    if mismatches == 0 and not state.killed:
        try:
            order_client.authorize_adoptions(tuple(authorization_candidates))
        except Exception as exc:  # noqa: BLE001 -- mutation authority must fail closed
            mismatches += 1
            reason = f"reconcile could not authorize exact broker orders: {exc}"
            order_client.revoke_adoptions()
            trip_kill(engine, reason, now=ts_now)
            await _send(notify, f"🛑 KILL SWITCH: {reason}")
        if (
            mismatches == 0
            and walk_md is not None
            and walk_tasks is not None
        ):
            resume = walk_resume if walk_resume is not None else resume_walks
            try:
                await resume(
                    engine=engine,
                    settings=settings,
                    order_client=order_client,
                    md=walk_md,
                    walk_tasks=walk_tasks,
                    notify=notify,
                )
            except asyncio.CancelledError:
                order_client.revoke_adoptions()
                raise
            except Exception as exc:  # noqa: BLE001 -- unmanaged working order halts
                log.exception("reconcile: walk resume failed")
                mismatches += 1
                reason = f"reconcile persisted price-walk resume failed: {exc}"
                order_client.revoke_adoptions()
                trip_kill(engine, reason, now=ts_now)
                await _send(notify, f"🛑 KILL SWITCH: {reason}")

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
