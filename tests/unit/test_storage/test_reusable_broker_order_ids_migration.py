"""Migration contract for session-local IBKR order IDs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import insert, update
from sqlalchemy.exc import IntegrityError

from alembic import command
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import orders


def test_0021_allows_terminal_id_reuse_but_rejects_two_active_owners(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "reusable-broker-id.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0020")

    engine = create_engine_for_path(db_path)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        prior_id = conn.execute(
            insert(orders).values(
                intent="close",
                symbol="NVDA",
                strategy="iron_condor",
                legs_json=[],
                quantity=1,
                ib_order_id=432,
                ib_perm_id=1264032972,
                order_ref="obot-45",
                status="cancelled",
                staged_ts=now,
                terminal_ts=now,
            )
        ).inserted_primary_key[0]
        current_id = conn.execute(
            insert(orders).values(
                intent="open",
                symbol="USO",
                strategy="iron_butterfly",
                legs_json=[],
                quantity=1,
                order_ref="obot-66",
                status="submitting",
                staged_ts=now,
            )
        ).inserted_primary_key[0]
    engine.dispose()

    command.upgrade(cfg, "0021")
    upgraded = create_engine_for_path(db_path)
    with upgraded.begin() as conn:
        conn.execute(
            update(orders)
            .where(orders.c.id == current_id)
            .values(status="submitted", ib_order_id=432, ib_perm_id=1778777000)
        )
        rows = conn.execute(
            orders.select().where(orders.c.id.in_([prior_id, current_id]))
        ).fetchall()
        assert [row.ib_order_id for row in rows] == [432, 432]

    with pytest.raises(IntegrityError):
        with upgraded.begin() as conn:
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="vertical",
                    legs_json=[],
                    quantity=1,
                    ib_order_id=432,
                    order_ref="obot-67",
                    status="submitted",
                    staged_ts=now,
                )
            )
    upgraded.dispose()
