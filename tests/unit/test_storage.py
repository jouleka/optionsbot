"""Tests for SQLite schema and connection setup."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, text

from optionsbot.storage import schema
from optionsbot.storage.db import create_engine_for_path

EXPECTED_TABLES = {
    "watchlist",
    "snapshots",
    "strategy_scores",
    "alerts",
    "scan_runs",
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _apply_migrations(db_path: Path) -> None:
    """Run alembic upgrade head against db_path using the Alembic Python API."""
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    # env.py honors sqlalchemy.url when present (see Task 4 Step 5), so we can
    # override the DB path without touching environment variables.
    command.upgrade(cfg, "head")


def test_metadata_lists_all_expected_tables() -> None:
    assert set(schema.metadata.tables.keys()) == EXPECTED_TABLES


def test_alerts_table_has_retry_queue_columns() -> None:
    cols = {c.name for c in schema.metadata.tables["alerts"].columns}
    assert {"status", "retry_count", "next_retry_ts", "last_error"} <= cols


def test_strategy_scores_has_snapshot_fk() -> None:
    t = schema.metadata.tables["strategy_scores"]
    fk_cols = {c.name for c in t.columns if c.foreign_keys}
    assert "snapshot_id" in fk_cols


def test_migration_applies_cleanly_and_creates_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        present = set(inspect(conn).get_table_names())
    assert EXPECTED_TABLES <= present


def test_wal_mode_is_set_on_connect(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal"


def test_can_insert_into_watchlist(tmp_path: Path) -> None:
    db_path = tmp_path / "insert.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.begin() as conn:
        conn.execute(
            schema.watchlist.insert().values(
                symbol="SPY",
                view_override_dir=None,
                view_override_iv=None,
                notes="initial watchlist entry",
                added_at=datetime(2026, 5, 26, 0, 0, 0),
            )
        )
        rows = conn.execute(schema.watchlist.select()).fetchall()
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"
