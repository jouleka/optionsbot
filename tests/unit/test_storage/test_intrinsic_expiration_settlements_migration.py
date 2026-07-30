"""Migration contract for intrinsic-value option expiration settlements."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from alembic import command
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import orders, position_settlements


def test_0022_preserves_worthless_and_allows_intrinsic_settlements(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "intrinsic-expiration.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0021")

    now = datetime.now(UTC)
    engine = create_engine_for_path(db_path)
    with engine.begin() as conn:
        first = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="vertical",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=now,
                    terminal_ts=now,
                )
            ).inserted_primary_key[0]
        )
        second = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="NVDA",
                    strategy="iron_butterfly",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=now,
                    terminal_ts=now,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(position_settlements).values(
                entry_order_id=first,
                kind="expired_worthless",
                expiry="20260729",
                terminal_spot=630.0,
                pnl=50.0,
                commissions=2.0,
                settled_at=now,
            )
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(position_settlements).values(
                    entry_order_id=second,
                    kind="expired_intrinsic",
                    expiry="20260729",
                    terminal_spot=190.01,
                    pnl=18.50,
                    commissions=2.49,
                    settled_at=now,
                )
            )
    engine.dispose()

    command.upgrade(cfg, "0022")
    upgraded = create_engine_for_path(db_path)
    with upgraded.begin() as conn:
        conn.execute(
            insert(position_settlements).values(
                entry_order_id=second,
                kind="expired_intrinsic",
                expiry="20260729",
                terminal_spot=190.01,
                pnl=18.50,
                commissions=2.49,
                settled_at=now,
            )
        )
        rows = conn.execute(
            select(
                position_settlements.c.entry_order_id,
                position_settlements.c.kind,
            ).order_by(position_settlements.c.entry_order_id)
        ).all()
    upgraded.dispose()

    assert rows == [
        (first, "expired_worthless"),
        (second, "expired_intrinsic"),
    ]
