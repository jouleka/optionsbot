"""Tests for SQLite schema and connection setup."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
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


def test_foreign_keys_pragma_is_on(tmp_path: Path) -> None:
    db_path = tmp_path / "fk.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        fk_on = conn.execute(text("PRAGMA foreign_keys")).scalar()
    assert fk_on == 1


def test_migration_creates_parent_directory(tmp_path: Path) -> None:
    # Regression: on a fresh machine the data directory does not exist yet.
    # `alembic upgrade head` must create it rather than failing with
    # "unable to open database file".
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    assert not nested.exists()
    db_path = nested / "first_run.db"
    _apply_migrations(db_path)
    assert db_path.exists()
    assert nested.is_dir()


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
                added_at=datetime(2026, 5, 26, 0, 0, 0, tzinfo=UTC),
            )
        )
        rows = conn.execute(schema.watchlist.select()).fetchall()
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"


def test_watchlist_view_override_dir_check_constraint(tmp_path: Path) -> None:
    db_path = tmp_path / "ck1.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    from sqlalchemy.exc import IntegrityError
    with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                schema.watchlist.insert().values(
                    symbol="SPY",
                    view_override_dir="bullish",  # invalid; not in enum
                    view_override_iv=None,
                    notes=None,
                    added_at=datetime(2026, 5, 26, 0, 0, 0, tzinfo=UTC),
                )
            )


def test_snapshots_regime_iv_check_constraint(tmp_path: Path) -> None:
    db_path = tmp_path / "ck2.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    from sqlalchemy.exc import IntegrityError
    with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                schema.snapshots.insert().values(
                    symbol="SPY",
                    ts=datetime(2026, 5, 26, tzinfo=UTC),
                    regime_iv="MEGA-HIGH",  # invalid
                )
            )


def test_snapshot_raw_json_roundtrips_as_dict(tmp_path: Path) -> None:
    # Demonstrates the JSON column accepts a dict directly and returns one
    # (no manual json.dumps/loads on the caller side).
    db_path = tmp_path / "json.db"
    _apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    payload = {"iv_rank": 0.42, "tags": ["high-iv", "neutral"]}
    with engine.begin() as conn:
        conn.execute(
            schema.snapshots.insert().values(
                symbol="SPY",
                ts=datetime(2026, 5, 26, tzinfo=UTC),
                raw_json=payload,
            )
        )
        row = conn.execute(schema.snapshots.select()).fetchone()
    assert row is not None
    assert row.raw_json == payload
