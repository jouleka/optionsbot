"""Net-liq (equity) drawdown circuit breaker (Phase 0, work-stream B).

Runs every order/exit tick: pulls live net liquidation, compares it to a
persisted day-start baseline, and trips the kill switch on a realized+unrealized
decline >= max_daily_loss_pct. A separate ``new_entry_allowed`` lets the entry
path stop ADDING risk once the drawdown reaches entry_block_loss_frac of the cap,
before the hard kill. The realized-only close-fill trip in order_watcher stays as
a backstop.

The baseline lives on the singleton execution_state row (id=1) so it survives an
intraday daemon restart; ``capture_day_start_net_liq`` is idempotent for a given
session date (set once, never overwritten downward).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Engine, select

from optionsbot.config import Settings
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.storage.schema import execution_state

log = logging.getLogger(__name__)

_ROW_ID = 1


@dataclass(frozen=True, slots=True)
class EquityVerdict:
    tripped: bool       # True iff THIS call newly tripped the kill switch
    evaluable: bool     # False when baseline or current net-liq is unknown
    drawdown_pct: float  # fractional decline from day-start (0.0 when not evaluable)
    reason: str


@dataclass(frozen=True, slots=True)
class EntryDecision:
    allowed: bool
    reason: str


def _day_start_row(engine: Engine) -> tuple[float | None, str | None]:
    with engine.connect() as conn:
        row = conn.execute(
            select(
                execution_state.c.day_start_net_liq,
                execution_state.c.day_start_session,
            ).where(execution_state.c.id == _ROW_ID)
        ).first()
    if row is None:
        return None, None
    return row.day_start_net_liq, row.day_start_session


def capture_day_start_net_liq(
    engine: Engine, net_liq: float, *, session: str | None = None
) -> float:
    """Persist the day-start net-liq baseline for ``session`` (idempotent).

    If a baseline already exists FOR THE SAME session, it is returned unchanged
    so an intraday restart can't reset the high-water-mark and hide a loss. A
    new session (or a fresh row) overwrites. ``session`` defaults to None for
    tests that don't care about the boundary; B2 passes the NYSE session date.
    """
    existing_nl, existing_session = _day_start_row(engine)
    if existing_nl is not None and existing_session == session:
        return float(existing_nl)
    # Ensure the singleton row exists, then set the baseline. We never disturb
    # killed/reason/ts here (the kill switch owns those).
    with engine.begin() as conn:
        updated = conn.execute(
            execution_state.update()
            .where(execution_state.c.id == _ROW_ID)
            .values(day_start_net_liq=net_liq, day_start_session=session)
        )
        if updated.rowcount == 0:
            conn.execute(
                execution_state.insert().values(
                    id=_ROW_ID, killed=0, reason=None, ts=None,
                    day_start_net_liq=net_liq, day_start_session=session,
                )
            )
    return net_liq


def _drawdown(day_start: float | None, current: float | None) -> float | None:
    if day_start is None or current is None or day_start <= 0:
        return None
    return (day_start - current) / day_start


def evaluate_net_liq_drawdown(
    engine: Engine,
    settings: Settings,
    *,
    current_net_liq: float | None,
    now: datetime,
) -> EquityVerdict:
    """Trip the kill switch on a >= max_daily_loss_pct day-start drawdown.

    Idempotent: if already killed, reports tripped=False and leaves the reason.
    """
    if load_state(engine).killed:
        return EquityVerdict(False, True, 0.0, "already killed")
    day_start, _ = _day_start_row(engine)
    dd = _drawdown(day_start, current_net_liq)
    if dd is None:
        return EquityVerdict(False, False, 0.0, "net-liq drawdown not evaluable")
    cap = settings.execution.max_daily_loss_pct
    if dd >= cap:
        reason = (
            f"net liq drawdown {dd * 100:.2f}% >= {cap * 100:.0f}% cap "
            f"(day-start ${day_start:,.0f} -> ${current_net_liq:,.0f})"
        )
        trip_kill(engine, reason, now=now)
        return EquityVerdict(True, True, dd, reason)
    return EquityVerdict(False, True, dd, f"net-liq drawdown {dd * 100:.2f}% under cap")


def new_entry_allowed(
    engine: Engine, settings: Settings, *, current_net_liq: float | None
) -> EntryDecision:
    """Block NEW entries once the drawdown reaches entry_block_loss_frac of the
    cap. Fails OPEN when not evaluable: the per-tick breaker is the real
    backstop, and a single flaky net-liq read shouldn't hard-block trading.
    """
    day_start, _ = _day_start_row(engine)
    dd = _drawdown(day_start, current_net_liq)
    if dd is None:
        return EntryDecision(True, "net-liq drawdown not evaluable (entry allowed)")
    cap = settings.execution.max_daily_loss_pct
    block_at = settings.execution.entry_block_loss_frac * cap
    if dd >= block_at:
        return EntryDecision(
            False,
            f"net-liq drawdown {dd * 100:.2f}% has reached the entry-block "
            f"threshold ({block_at * 100:.2f}% = {settings.execution.entry_block_loss_frac:.0%} "
            f"of the {cap * 100:.0f}% daily cap) — not adding new risk",
        )
    return EntryDecision(True, f"net-liq drawdown {dd * 100:.2f}% under entry-block threshold")
