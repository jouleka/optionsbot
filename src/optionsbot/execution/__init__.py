"""Execution layer (IBK-123+): arming gate + persisted kill switch.

This package contains NO order-placement code yet — IBK-123 ships only the
safety substrate (settings, paper-only interlock, kill switch) that the later
execution phases (orders schema IBK-124, OrderClient IBK-125, Telegram
/execute IBK-126, ...) sit behind.
"""

from optionsbot.execution.gate import PAPER_PORTS, GateResult, can_execute
from optionsbot.execution.orders import (
    LEGAL_TRANSITIONS,
    ORDER_STATUSES,
    TERMINAL_STATUSES,
    WORKING_STATUSES,
    IllegalOrderTransition,
    OrderRecord,
    bump_reprice,
    get_order,
    net_premium,
    open_orders,
    record_fill,
    set_fill_commission,
    stage_order,
    transition,
    working_orders,
)
from optionsbot.execution.state import (
    ExecutionState,
    clear_kill,
    load_state,
    trip_kill,
)

__all__ = [
    "LEGAL_TRANSITIONS",
    "ORDER_STATUSES",
    "PAPER_PORTS",
    "TERMINAL_STATUSES",
    "WORKING_STATUSES",
    "ExecutionState",
    "GateResult",
    "IllegalOrderTransition",
    "OrderRecord",
    "bump_reprice",
    "can_execute",
    "clear_kill",
    "get_order",
    "load_state",
    "net_premium",
    "open_orders",
    "record_fill",
    "set_fill_commission",
    "stage_order",
    "transition",
    "trip_kill",
    "working_orders",
]
