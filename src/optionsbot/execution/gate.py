"""Pure arming gate for order execution (IBK-123).

``can_execute`` is THE decision point every order-placing path must pass
before touching the IBKR order API. Deny precedence (loudest first):
paper-only interlock -> kill switch -> execution.enabled. A live-port
misconfiguration must out-shout a mere "execution disabled".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from optionsbot.config import PAPER_PORTS

if TYPE_CHECKING:
    from optionsbot.config import Settings
    from optionsbot.execution.state import ExecutionState

__all__ = ["PAPER_PORTS", "GateResult", "can_execute", "can_reduce_risk"]


@dataclass(frozen=True, slots=True)
class GateResult:
    allowed: bool
    reason: str


def _base_interlock(settings: Settings) -> GateResult | None:
    """Return a non-kill execution denial shared by entry and exit gates."""
    execution = settings.execution
    if execution.paper_only:
        if not settings.ibkr.paper:
            return GateResult(False, "paper-only interlock: ibkr.paper is false (live account?)")
        if settings.ibkr.port not in PAPER_PORTS:
            return GateResult(
                False,
                f"paper-only interlock: port {settings.ibkr.port} is not a "
                "recognized paper port (4002 Gateway / 7497 TWS)",
            )
    if not execution.enabled:
        return GateResult(False, "execution.enabled is false (analysis/alerting only)")
    return None


def can_reduce_risk(settings: Settings) -> GateResult:
    """Allow protective closes while preserving account/config interlocks.

    A kill switch means "add no new risk". It must never strand an existing
    position by disabling deterministic stops, expiry flattening, or a human
    close request. Paper/live and execution-enabled interlocks still apply.
    """
    denial = _base_interlock(settings)
    if denial is not None:
        return denial
    return GateResult(
        True,
        "risk-reducing execution allowed (paper)"
        if settings.execution.paper_only
        else "risk-reducing execution allowed (PAPER-ONLY INTERLOCK OFF)",
    )


def can_execute(settings: Settings, state: ExecutionState) -> GateResult:
    """Decide whether order placement is currently permitted.

    Pure function of (settings, persisted kill-switch state) so the full
    truth table is unit-testable; callers load the state via
    ``optionsbot.execution.state.load_state``.
    """
    execution = settings.execution
    interlock_denial = _base_interlock(settings)
    # Preserve the original deny precedence: paper/live outranks kill, and kill
    # outranks a merely disabled engine.
    if interlock_denial is not None and "paper-only interlock" in interlock_denial.reason:
        return interlock_denial
    if state.killed:
        return GateResult(False, f"kill switch tripped: {state.reason or 'no reason recorded'}")
    if interlock_denial is not None:
        return interlock_denial
    return GateResult(
        True,
        "armed (paper)" if execution.paper_only else "armed (PAPER-ONLY INTERLOCK OFF)",
    )
