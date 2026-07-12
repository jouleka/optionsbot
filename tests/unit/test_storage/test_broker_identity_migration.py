"""Migration contract for unique broker-order ownership."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from alembic.config import Config
from sqlalchemy import insert, inspect, select

from alembic import command
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import execution_state, orders


def test_0017_quarantines_duplicate_broker_order_owners(tmp_path: Path) -> None:
    db_path = tmp_path / "broker-identity.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0016")

    engine = create_engine_for_path(db_path)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        for ref in ("obot-1", "obot-2"):
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="vertical",
                    legs_json=[],
                    quantity=1,
                    ib_order_id=77,
                    order_ref=ref,
                    status="submitted",
                    staged_ts=now,
                )
            )
    engine.dispose()

    command.upgrade(cfg, "0017")
    upgraded = create_engine_for_path(db_path)
    indexes = {row["name"]: row for row in inspect(upgraded).get_indexes("orders")}
    assert indexes["uq_orders_ib_order_id"]["unique"]
    with upgraded.connect() as conn:
        rows = conn.execute(select(orders).order_by(orders.c.id)).fetchall()
        state = conn.execute(select(execution_state)).one()
        assert [row.ib_order_id for row in rows] == [None, None]
        assert all("duplicate broker order id 77" in row.last_error for row in rows)
        assert state.killed
        assert "ambiguous broker order ownership" in state.reason
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    upgraded.dispose()

    command.downgrade(cfg, "0016")
    downgraded = create_engine_for_path(db_path)
    assert "uq_orders_ib_order_id" not in {
        row["name"] for row in inspect(downgraded).get_indexes("orders")
    }
    downgraded.dispose()

    command.upgrade(cfg, "0017")
    reupgraded = create_engine_for_path(db_path)
    assert "uq_orders_ib_order_id" in {
        row["name"] for row in inspect(reupgraded).get_indexes("orders")
    }
    with reupgraded.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
