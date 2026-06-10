"""Pure arming gate for order execution (IBK-123).

``can_execute`` is THE decision point every order-placing path must pass
before touching the IBKR order API. Deny precedence (loudest first):
paper-only interlock -> kill switch -> execution.enabled. A live-port
misconfiguration must out-shout a mere "execution disabled".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optionsbot.config import Settings
    from optionsbot.execution.state import ExecutionState

# Recognized paper ports: IB Gateway paper / TWS paper (IBK-16 conventions).
PAPER_PORTS: frozenset[int] = frozenset({4002, 7497})


@dataclass(frozen=True, slots=True)
class GateResult:
    allowed: bool
    reason: str


def can_execute(settings: Settings, state: ExecutionState) -> GateResult:
    """Decide whether order placement is currently permitted.

    Pure function of (settings, persisted kill-switch state) so the full
    truth table is unit-testable; callers load the state via
    ``optionsbot.execution.state.load_state``.
    """
    execution = settings.execution
    if execution.paper_only:
        if not settings.ibkr.paper:
            return GateResult(
                False, "paper-only interlock: ibkr.paper is false (live account?)"
            )
        if settings.ibkr.port not in PAPER_PORTS:
            return GateResult(
                False,
                f"paper-only interlock: port {settings.ibkr.port} is not a "
                "recognized paper port (4002 Gateway / 7497 TWS)",
            )
    if state.killed:
        return GateResult(
            False, f"kill switch tripped: {state.reason or 'no reason recorded'}"
        )
    if not execution.enabled:
        return GateResult(False, "execution.enabled is false (analysis/alerting only)")
    return GateResult(
        True,
        "armed (paper)" if execution.paper_only else "armed (PAPER-ONLY INTERLOCK OFF)",
    )
