"""Tests for SQLite schema and connection setup."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text

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
    "pick_outcomes",
    "symbol_news",
    "position_alerts",
    "execution_state",
    "hermes_overlay_state",
    "orders",
    "fills",
    "order_quotes",
    "walk_state",
    "exit_requests",
    "entry_reviews",
    "entry_intent_consumptions",
    "position_exit_state",
    "position_settlements",
    "managed_opportunities",
    "managed_opportunity_marks",
    "managed_context_reviews",
    "managed_models",
    "managed_model_evaluations",
}


def test_metadata_lists_all_expected_tables() -> None:
    assert set(schema.metadata.tables.keys()) == EXPECTED_TABLES


def test_position_alerts_migration_round_trips(tmp_path: Path) -> None:
    """0006 upgrade->downgrade->upgrade is reversible (guards a future downgrade edit)."""
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "roundtrip.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    def tables() -> set[str]:
        with create_engine_for_path(db).connect() as conn:
            return set(inspect(conn).get_table_names())

    command.upgrade(cfg, "head")
    assert "position_alerts" in tables()
    # Target 0005 explicitly (not "-1"): head keeps moving as migrations land,
    # but this test pins 0006's own downgrade path.
    command.downgrade(cfg, "0005")
    assert "position_alerts" not in tables()
    command.upgrade(cfg, "head")
    assert "position_alerts" in tables()


def test_execution_state_migration_round_trips(tmp_path: Path) -> None:
    """0007 upgrade->downgrade->upgrade is reversible (guards a future downgrade edit)."""
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "roundtrip_exec.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    def tables() -> set[str]:
        with create_engine_for_path(db).connect() as conn:
            return set(inspect(conn).get_table_names())

    command.upgrade(cfg, "head")
    assert "execution_state" in tables()
    # Target 0006 explicitly (not "-1") so the test survives future heads.
    command.downgrade(cfg, "0006")
    assert "execution_state" not in tables()
    command.upgrade(cfg, "head")
    assert "execution_state" in tables()


def test_orders_fills_migration_round_trips(tmp_path: Path) -> None:
    """0008 upgrade->downgrade->upgrade is reversible (guards a future downgrade edit)."""
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "roundtrip_orders.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    def tables() -> set[str]:
        with create_engine_for_path(db).connect() as conn:
            return set(inspect(conn).get_table_names())

    command.upgrade(cfg, "head")
    assert {"orders", "fills"} <= tables()
    # Target 0007 explicitly (not "-1") so the test survives future heads.
    command.downgrade(cfg, "0007")
    assert {"orders", "fills"} & tables() == set()
    command.upgrade(cfg, "head")
    assert {"orders", "fills"} <= tables()


def test_orders_closes_column_exists_after_migration(tmp_path: Path) -> None:
    """0010 adds orders.closes_order_id (IBK-129 close↔entry linkage)."""
    db_path = tmp_path / "closes.db"
    apply_migrations(db_path)
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("orders")}
    assert "closes_order_id" in cols


def test_active_close_unique_index_migration_round_trips(tmp_path: Path) -> None:
    from alembic.config import Config
    from sqlalchemy.exc import IntegrityError

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "roundtrip_active_close.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    engine = create_engine_for_path(db)

    def index_names() -> set[str]:
        with engine.connect() as conn:
            return {idx["name"] for idx in inspect(conn).get_indexes("orders")}

    command.upgrade(cfg, "head")
    assert "uq_orders_active_close_per_entry" in index_names()
    with engine.begin() as conn:
        entry_id = int(conn.execute(schema.orders.insert().values(
            intent="open", symbol="SPY", strategy="bull_put_spread", legs_json=[],
            quantity=1, status="filled", staged_ts=datetime.now(UTC), reprice_count=0,
        )).inserted_primary_key[0])
        conn.execute(schema.orders.insert().values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy="bull_put_spread", legs_json=[], quantity=1, status="staged",
            staged_ts=datetime.now(UTC), reprice_count=0,
        ))
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(schema.orders.insert().values(
                intent="close", closes_order_id=entry_id, symbol="SPY",
                strategy="bull_put_spread", legs_json=[], quantity=1, status="submitting",
                staged_ts=datetime.now(UTC), reprice_count=0,
            ))

    command.downgrade(cfg, "0014")
    assert "uq_orders_active_close_per_entry" not in index_names()
    with engine.begin() as conn:
        duplicate_id = int(conn.execute(schema.orders.insert().values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy="bull_put_spread", legs_json=[], quantity=1, status="submitted",
            staged_ts=datetime.now(UTC), reprice_count=0,
        )).inserted_primary_key[0])
        conn.execute(
            schema.orders.update().where(schema.orders.c.id == duplicate_id).values(
                status="skipped", terminal_ts=datetime.now(UTC)
            )
        )
    command.upgrade(cfg, "head")
    assert "uq_orders_active_close_per_entry" in index_names()


def test_active_close_migration_quarantines_legacy_duplicate_claims(
    tmp_path: Path,
) -> None:
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "duplicate_active_close.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "0015")
    engine = create_engine_for_path(db)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        entry_id = int(
            conn.execute(
                schema.orders.insert().values(
                    intent="open",
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=now,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
        for status in ("staged", "submitted"):
            conn.execute(
                schema.orders.insert().values(
                    intent="close",
                    closes_order_id=entry_id,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=[],
                    quantity=1,
                    status=status,
                    staged_ts=now,
                    reprice_count=0,
                )
            )

    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        closes = conn.execute(
            schema.orders.select()
            .where(schema.orders.c.closes_order_id == entry_id)
            .order_by(schema.orders.c.id)
        ).fetchall()
        state = conn.execute(
            schema.execution_state.select().where(schema.execution_state.c.id == 1)
        ).one()
        revision = conn.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
        indexes = {idx["name"] for idx in inspect(conn).get_indexes("orders")}
    assert [row.status for row in closes] == ["abandoned", "submitted"]
    assert closes[0].terminal_ts is not None
    assert "migration 0016" in closes[0].last_error
    assert state.killed == 1
    assert "duplicate active closes" in state.reason
    assert revision == "0023"
    assert "uq_orders_active_close_per_entry" in indexes
    assert "uq_orders_ib_order_id" in indexes

    command.downgrade(cfg, "0015")
    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        statuses = conn.execute(
            select(schema.orders.c.status)
            .where(schema.orders.c.closes_order_id == entry_id)
            .order_by(schema.orders.c.id)
        ).scalars().all()
        state_after_roundtrip = conn.execute(
            schema.execution_state.select().where(schema.execution_state.c.id == 1)
        ).one()
        revision_after_roundtrip = conn.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
    assert statuses == ["abandoned", "submitted"]
    assert state_after_roundtrip.killed == 1
    assert revision_after_roundtrip == "0023"


def test_order_quotes_migration_round_trips(tmp_path: Path) -> None:
    """0009 upgrade->downgrade->upgrade is reversible."""
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db = tmp_path / "roundtrip_quotes.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")

    def tables() -> set[str]:
        with create_engine_for_path(db).connect() as conn:
            return set(inspect(conn).get_table_names())

    command.upgrade(cfg, "head")
    assert "order_quotes" in tables()
    command.downgrade(cfg, "0008")
    assert "order_quotes" not in tables()
    command.upgrade(cfg, "head")
    assert "order_quotes" in tables()


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


def test_managed_capture_migration_upgrades_a_populated_0022_database(
    tmp_path: Path,
) -> None:
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    db_path = tmp_path / "populated-0022.db"
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0022")
    engine = create_engine_for_path(db_path)
    with engine.begin() as conn:
        conn.execute(
            schema.watchlist.insert().values(
                symbol="SPY",
                added_at=datetime(2026, 8, 28, tzinfo=UTC),
            )
        )
    engine.dispose()

    command.upgrade(cfg, "0023")

    upgraded = create_engine_for_path(db_path)
    inspector = inspect(upgraded)
    columns = {
        column["name"]
        for column in inspector.get_columns("managed_opportunities")
    }
    with upgraded.connect() as conn:
        symbols = conn.execute(select(schema.watchlist.c.symbol)).scalars().all()
        revision = conn.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
        integrity = conn.exec_driver_sql("PRAGMA integrity_check").scalar_one()
    upgraded.dispose()

    assert symbols == ["SPY"]
    assert {
        "decision_batch_id",
        "decision_score",
        "decision_defined_risk",
        "decision_max_loss",
        "decision_account_value_available",
        "decision_account_value_usd",
    } <= columns
    assert revision == "0023"
    assert integrity == "ok"


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
