"""Tests for SQLite schema and connection setup."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from optionsbot.storage import schema
from optionsbot.storage.db import create_engine_for_path
from tests.conftest import apply_migrations  # noqa: TID252 (cross-package import OK in tests)

EXPECTED_TABLES = {
    "watchlist",
    "snapshots",
    "strategy_scores",
    "alerts",
    "scan_runs",
    "iv_history",
}


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
    apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        present = set(inspect(conn).get_table_names())
    assert EXPECTED_TABLES <= present


def test_wal_mode_is_set_on_connect(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.db"
    apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal"


def test_foreign_keys_pragma_is_on(tmp_path: Path) -> None:
    db_path = tmp_path / "fk.db"
    apply_migrations(db_path)
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
    apply_migrations(db_path)
    assert db_path.exists()
    assert nested.is_dir()


def test_can_insert_into_watchlist(tmp_path: Path) -> None:
    db_path = tmp_path / "insert.db"
    apply_migrations(db_path)
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
    apply_migrations(db_path)
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
    apply_migrations(db_path)
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
    apply_migrations(db_path)
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


def test_record_atm_iv_upserts_latest_per_day(tmp_path: Path) -> None:
    from datetime import date

    from optionsbot.storage.iv_history import read_atm_iv_history, record_atm_iv

    db_path = tmp_path / "ivh.db"
    apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    d = date(2026, 5, 26)
    record_atm_iv(engine, "SPY", d, 0.20)
    record_atm_iv(engine, "SPY", d, 0.25)  # same (symbol, date) -> update, not duplicate
    series = read_atm_iv_history(engine, "SPY")
    assert list(series) == [0.25]


def test_read_atm_iv_history_orders_oldest_to_newest(tmp_path: Path) -> None:
    from datetime import date

    from optionsbot.storage.iv_history import read_atm_iv_history, record_atm_iv

    db_path = tmp_path / "ivh2.db"
    apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    record_atm_iv(engine, "SPY", date(2026, 5, 27), 0.22)
    record_atm_iv(engine, "SPY", date(2026, 5, 25), 0.18)
    record_atm_iv(engine, "SPY", date(2026, 5, 26), 0.20)
    series = read_atm_iv_history(engine, "SPY")
    assert list(series) == [0.18, 0.20, 0.22]


def test_read_atm_iv_history_unknown_symbol_is_empty(tmp_path: Path) -> None:
    from optionsbot.storage.iv_history import read_atm_iv_history

    db_path = tmp_path / "ivh3.db"
    apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    series = read_atm_iv_history(engine, "NOPE")
    assert len(series) == 0
