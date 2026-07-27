"""Durable high-water marks for adaptive position exits."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Engine, insert, select, update

from optionsbot.storage.schema import position_exit_state


def peak_pnl_for(engine: Engine, entry_order_id: int) -> float | None:
    """Return the persisted per-unit P&L peak for an entry, if observed."""
    with engine.connect() as conn:
        value = conn.execute(
            select(position_exit_state.c.peak_pnl_per_unit).where(
                position_exit_state.c.entry_order_id == entry_order_id
            )
        ).scalar_one_or_none()
    return None if value is None else float(value)


def observe_pnl(
    engine: Engine,
    entry_order_id: int,
    pnl_per_unit: float,
    *,
    now: datetime,
) -> float:
    """Persist and return the greater of the prior and current P&L.

    Exit ticks are serialized by the daemon, so a small select/update
    transaction is sufficient and remains portable across SQLite/Postgres.
    """
    with engine.begin() as conn:
        prior = conn.execute(
            select(position_exit_state.c.peak_pnl_per_unit).where(
                position_exit_state.c.entry_order_id == entry_order_id
            )
        ).scalar_one_or_none()
        peak = pnl_per_unit if prior is None else max(float(prior), pnl_per_unit)
        if prior is None:
            conn.execute(
                insert(position_exit_state).values(
                    entry_order_id=entry_order_id,
                    peak_pnl_per_unit=peak,
                    updated_at=now,
                )
            )
        else:
            conn.execute(
                update(position_exit_state)
                .where(position_exit_state.c.entry_order_id == entry_order_id)
                .values(peak_pnl_per_unit=peak, updated_at=now)
            )
    return peak
