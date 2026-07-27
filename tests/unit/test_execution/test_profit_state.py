"""Durable adaptive-exit high-water mark tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, insert

from optionsbot.execution.profit_state import observe_pnl, peak_pnl_for
from optionsbot.storage.schema import orders

NOW = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)


def _entry(engine: Engine) -> int:
    with engine.begin() as conn:
        return int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="QQQ",
                    strategy="bear_put_spread",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=NOW,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )


def test_observe_pnl_preserves_peak_across_reads(tmp_db: Engine) -> None:
    entry_id = _entry(tmp_db)

    assert peak_pnl_for(tmp_db, entry_id) is None
    assert observe_pnl(tmp_db, entry_id, 0.69, now=NOW) == 0.69
    assert observe_pnl(
        tmp_db,
        entry_id,
        0.40,
        now=NOW + timedelta(minutes=1),
    ) == 0.69
    assert observe_pnl(
        tmp_db,
        entry_id,
        1.10,
        now=NOW + timedelta(minutes=2),
    ) == 1.10
    assert peak_pnl_for(tmp_db, entry_id) == 1.10
