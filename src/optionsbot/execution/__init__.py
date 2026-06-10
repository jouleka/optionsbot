"""Execution layer (IBK-123+): arming gate + persisted kill switch.

This package contains NO order-placement code yet — IBK-123 ships only the
safety substrate (settings, paper-only interlock, kill switch) that the later
execution phases (orders schema IBK-124, OrderClient IBK-125, Telegram
/execute IBK-126, ...) sit behind.
"""

from optionsbot.execution.gate import PAPER_PORTS, GateResult, can_execute
from optionsbot.execution.state import (
    ExecutionState,
    clear_kill,
    load_state,
    trip_kill,
)

__all__ = [
    "PAPER_PORTS",
    "ExecutionState",
    "GateResult",
    "can_execute",
    "clear_kill",
    "load_state",
    "trip_kill",
]
