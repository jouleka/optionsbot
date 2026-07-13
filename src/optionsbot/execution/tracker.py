"""OrderTracker — binds OrderClient events to the IBK-124 ledger (IBK-125).

Handlers run synchronously on the ib_async event loop, so they must NEVER
raise: terminal re-delivery, unknown order_refs, and duplicate executions are
all expected broker traffic and are logged-and-ignored. SQLite writes are
local and fast (ms) — acceptable on the loop.

Deliberately NOT re-exported from ``optionsbot.execution.__init__``: this
module imports ``optionsbot.ibkr.types`` while ``optionsbot.ibkr.orders``
imports config-level constants shared with the execution gate — keeping the
tracker out of the package init keeps the import graph acyclic. Import it
explicitly: ``from optionsbot.execution.tracker import OrderTracker``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from optionsbot.execution.orders import (
    FAILED_TERMINAL_STATUSES,
    IllegalOrderTransition,
    get_order,
    record_fill,
    set_fill_commission,
    transition,
)
from optionsbot.execution.state import trip_kill
from optionsbot.ibkr.types import (
    CommissionUpdate,
    ExecutionFill,
    OrderStatusUpdate,
    ledger_row_id_from_ref,
)
from optionsbot.storage.schema import orders

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from optionsbot.ibkr.orders import BrokerCallbackKind, OrderClient

log = logging.getLogger(__name__)

_ORDER_REF_PREFIX = "obot-"

# Transient states the ledger doesn't model — ignored on purpose.
_IGNORED_STATUSES = frozenset(
    {"PendingSubmit", "PendingCancel", "ApiPending", "ApiUpdate", "ValidationError"}
)


def map_ib_status(status: str, filled: float, remaining: float) -> str | None:
    """IBKR orderStatus string -> ledger status (None = ignore this event)."""
    if status in _IGNORED_STATUSES:
        return None
    if status in ("PreSubmitted", "Submitted"):
        return "partial" if filled > 0 else "submitted"
    if status == "Filled":
        return "filled"
    if status in ("Cancelled", "ApiCancelled"):
        return "cancelled"
    if status == "Inactive":
        # No native 'expired' status (deliberate gap, see the 0008 plan doc);
        # Inactive = IBKR deactivated/rejected the (remaining) order.
        return "rejected"
    log.warning("unrecognized IBKR order status %r — ignored", status)
    return None


def row_id_from_ref(order_ref: str | None) -> int | None:
    """'obot-123' -> 123; anything else (manual trades, None) -> None."""
    return ledger_row_id_from_ref(order_ref)


class OrderTracker:
    """Persists OrderClient events into the orders/fills ledger."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def attach(self, order_client: OrderClient) -> None:
        order_client.on_callback_error(self.handle_callback_error)
        order_client.on_status(self.handle_status)
        order_client.on_fill(self.handle_fill)
        order_client.on_commission(self.handle_commission)

    def handle_callback_error(
        self, kind: BrokerCallbackKind, error: Exception
    ) -> None:
        reason = f"live broker {kind} callback failed validation or handling"
        try:
            trip_kill(self._engine, reason)
        except Exception:
            log.critical(
                "failed to persist kill switch after broker %s callback failure",
                kind,
                exc_info=True,
            )
        log.error(reason, exc_info=(type(error), error, error.__traceback__))

    def handle_status(self, update: OrderStatusUpdate) -> None:
        row_id = row_id_from_ref(update.order_ref)
        if row_id is None:
            return  # not one of ours (manual trade or foreign client)
        record = get_order(self._engine, row_id)
        if record is None:
            trip_kill(
                self._engine,
                f"live broker status for bot order #{row_id} missing from ledger",
            )
            log.error("KILL: live status for unknown ledger order #%s", row_id)
            return
        with self._engine.connect() as conn:
            existing_owner = conn.execute(
                select(orders.c.id).where(
                    orders.c.ib_order_id == update.ib_order_id,
                    orders.c.id != row_id,
                )
            ).first()
        if existing_owner is not None:
            trip_kill(
                self._engine,
                f"live broker order {update.ib_order_id} already belongs to "
                f"ledger order #{existing_owner.id}, cannot bind to #{row_id}",
            )
            log.error("KILL: duplicate live broker order identity %s", update.ib_order_id)
            return
        if record.ib_order_id is not None and record.ib_order_id != update.ib_order_id:
            trip_kill(
                self._engine,
                f"live broker order identity mismatch for #{row_id}: "
                f"ledger={record.ib_order_id}, broker={update.ib_order_id}",
            )
            log.error("KILL: live broker identity mismatch for order #%s", row_id)
            return
        if update.status in _IGNORED_STATUSES:
            return
        target = map_ib_status(update.status, update.filled, update.remaining)
        if target is None:
            trip_kill(
                self._engine,
                f"live broker status unknown for order #{row_id}: {update.status!r}",
            )
            log.error("KILL: unknown live broker status for order #%s", row_id)
            return
        error = "deactivated/rejected by IBKR (Inactive)" if target == "rejected" else None
        try:
            transition(
                self._engine,
                row_id,
                target,
                ib_order_id=update.ib_order_id,
                ib_perm_id=update.perm_id,
                error=error,
            )
        except IllegalOrderTransition:
            # Expected broker noise: e.g. 'Filled' re-delivered after the row
            # is terminal. The working-state self-loops are legal; anything
            # else here is re-delivery of a terminal state.
            log.debug(
                "ignored status re-delivery for order %s: %s", row_id, update.status
            )
        except ValueError as exc:
            trip_kill(
                self._engine,
                f"live broker status could not be persisted for order #{row_id}: {exc}",
            )
            log.error("KILL: orderStatus persistence failed for order %s", row_id)

    def handle_fill(self, fill: ExecutionFill) -> None:
        row_id = row_id_from_ref(fill.order_ref)
        if row_id is None:
            return
        if fill.sec_type not in {"OPT", "BAG"}:
            trip_kill(
                self._engine,
                f"live execution {fill.exec_id!r} has unknown security type "
                f"{fill.sec_type!r} for order #{row_id}",
            )
            return
        record = get_order(self._engine, row_id)
        if record is None:
            trip_kill(
                self._engine,
                f"live execution {fill.exec_id!r} references missing order #{row_id}",
            )
            log.error("KILL: live fill for unknown ledger order #%s", row_id)
            return
        if fill.sec_type == "BAG":
            # Per-LEG rows only: a BAG-level summary execution alongside the
            # leg executions would double-count net_premium.
            return
        try:
            recorded = record_fill(
                self._engine,
                row_id,
                exec_id=fill.exec_id,
                side=fill.side,
                price=fill.price,
                qty=fill.qty,
                ts=fill.ts,
                leg_con_id=fill.con_id,
            )
        except ValueError as exc:
            trip_kill(
                self._engine,
                f"live fill evidence invalid for order #{row_id}: {exc}",
            )
            log.error("KILL: invalid live fill for order %s", row_id)
            return
        if not recorded:
            log.debug("duplicate execId %s ignored (replay)", fill.exec_id)
            return
        # IBK-128: a LIVE fill for an order the ledger already wrote off means
        # the broker holds a position the books deny (e.g. a cancel raced a
        # fill) — stop everything, human required.
        record = get_order(self._engine, row_id)
        if record is not None and record.status in FAILED_TERMINAL_STATUSES:
            trip_kill(
                self._engine,
                f"live fill {fill.exec_id} for order #{row_id} already in "
                f"status {record.status} — ledger denies a real position",
            )
            log.error(
                "KILL: live fill %s on failed-terminal order #%s (%s)",
                fill.exec_id, row_id, record.status,
            )

    def handle_commission(self, update: CommissionUpdate) -> None:
        try:
            attached = set_fill_commission(
                self._engine, update.exec_id, update.commission
            )
        except ValueError as exc:
            trip_kill(
                self._engine,
                f"live commission evidence conflicts for {update.exec_id!r}: {exc}",
            )
            log.error("KILL: conflicting commission for %s", update.exec_id)
            return
        if not attached:
            log.debug("commission for unknown execId %s — ignored", update.exec_id)
