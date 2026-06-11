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

from optionsbot.execution.orders import (
    IllegalOrderTransition,
    record_fill,
    set_fill_commission,
    transition,
)
from optionsbot.ibkr.types import CommissionUpdate, ExecutionFill, OrderStatusUpdate

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from optionsbot.ibkr.orders import OrderClient

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
    if not order_ref or not order_ref.startswith(_ORDER_REF_PREFIX):
        return None
    suffix = order_ref[len(_ORDER_REF_PREFIX):]
    return int(suffix) if suffix.isdigit() else None


class OrderTracker:
    """Persists OrderClient events into the orders/fills ledger."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def attach(self, order_client: OrderClient) -> None:
        order_client.on_status(self.handle_status)
        order_client.on_fill(self.handle_fill)
        order_client.on_commission(self.handle_commission)

    def handle_status(self, update: OrderStatusUpdate) -> None:
        row_id = row_id_from_ref(update.order_ref)
        if row_id is None:
            return  # not one of ours (manual trade or foreign client)
        target = map_ib_status(update.status, update.filled, update.remaining)
        if target is None:
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
        except ValueError:
            log.warning("orderStatus for unknown ledger row %s — ignored", row_id)

    def handle_fill(self, fill: ExecutionFill) -> None:
        if fill.sec_type == "BAG":
            # Per-LEG rows only: a BAG-level summary execution alongside the
            # leg executions would double-count net_premium.
            return
        row_id = row_id_from_ref(fill.order_ref)
        if row_id is None:
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
        except ValueError:
            log.warning("fill with invalid side %r for order %s", fill.side, row_id)
            return
        if not recorded:
            log.debug("duplicate execId %s ignored (replay)", fill.exec_id)

    def handle_commission(self, update: CommissionUpdate) -> None:
        if not set_fill_commission(self._engine, update.exec_id, update.commission):
            log.debug("commission for unknown execId %s — ignored", update.exec_id)
