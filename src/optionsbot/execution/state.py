"""Persisted execution kill switch (IBK-123).

One singleton row (id=1) in ``execution_state``. The switch must survive
daemon restarts — a tripped kill that silently re-armed on redeploy would
defeat its purpose — hence DB-backed rather than a DaemonContext flag like
``alerting_paused``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, select

from optionsbot.storage.schema import execution_state

_ROW_ID = 1
_SESSION_LOSS_KILL_PREFIXES = (
    "net liq drawdown ",
    "daily realized loss ",
    "daily cumulative Hermes realized-loss cap breached ",
)
_CLEAN_RECONCILE_KILL_PREFIXES = (
    "price-walk modify outcome unknown for order #",
    "cancel request outcome unknown for order #",
)


@dataclass(frozen=True, slots=True)
class ExecutionState:
    killed: bool
    reason: str | None
    ts: datetime | None


def load_state(engine: Engine) -> ExecutionState:
    """Current switch state; a missing row means never tripped (not killed)."""
    with engine.connect() as conn:
        row = conn.execute(
            select(execution_state).where(execution_state.c.id == _ROW_ID)
        ).first()
    if row is None:
        return ExecutionState(killed=False, reason=None, ts=None)
    ts = row.ts
    if ts is not None and ts.tzinfo is None:
        # SQLite DateTime(timezone=True) drops tzinfo on read; values are
        # written as UTC, so re-attach it (same defense as alert_dedup).
        ts = ts.replace(tzinfo=UTC)
    return ExecutionState(killed=bool(row.killed), reason=row.reason, ts=ts)


def trip_kill(
    engine: Engine, reason: str, *, now: datetime | None = None
) -> ExecutionState:
    return _write(engine, killed=True, reason=reason, now=now)


def clear_kill(engine: Engine, *, now: datetime | None = None) -> ExecutionState:
    return _write(engine, killed=False, reason=None, now=now)


def is_session_loss_kill(reason: str | None) -> bool:
    """True only for loss limits whose authority ends with the NYSE session.

    Manual halts, reconciliation mismatches, broker uncertainty, non-atomic
    closes, and other operational kills intentionally remain latched until an
    explicit repair and re-arm.
    """
    if reason is None:
        return False
    return reason.startswith(_SESSION_LOSS_KILL_PREFIXES) or reason.endswith(
        " consecutive losing trades this session"
    )


def is_clean_reconcile_recoverable_kill(reason: str | None) -> bool:
    """Whether exact clean broker proof may recover a mutation uncertainty.

    This is deliberately narrow. Manual halts, risk breakers, position
    mismatches, callback failures, and arbitrary reconciliation errors remain
    latched for explicit review.
    """
    return reason is not None and reason.startswith(_CLEAN_RECONCILE_KILL_PREFIXES)


def _write(
    engine: Engine, *, killed: bool, reason: str | None, now: datetime | None
) -> ExecutionState:
    ts = now if now is not None else datetime.now(UTC)
    with engine.begin() as conn:
        updated = conn.execute(
            execution_state.update()
            .where(execution_state.c.id == _ROW_ID)
            .values(killed=int(killed), reason=reason, ts=ts)
        )
        if updated.rowcount == 0:
            conn.execute(
                execution_state.insert().values(
                    id=_ROW_ID, killed=int(killed), reason=reason, ts=ts
                )
            )
    return ExecutionState(killed=killed, reason=reason, ts=ts)
