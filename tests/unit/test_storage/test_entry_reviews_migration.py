"""Migration contract for Hermes entry-review requests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import Engine, delete, func, insert, inspect, select, update
from sqlalchemy.exc import IntegrityError

from alembic import command
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import (
    alerts,
    entry_intent_consumptions,
    entry_reviews,
    orders,
    snapshots,
    strategy_scores,
)


def test_entry_reviews_table_is_migrated(tmp_db: Engine) -> None:
    inspector = inspect(tmp_db)

    assert "entry_reviews" in inspector.get_table_names()
    columns = {column["name"] for column in inspector.get_columns("entry_reviews")}
    assert columns == {
        "id",
        "strategy_score_id",
        "alert_id",
        "reviewed_at",
        "verdict",
        "confidence",
        "sources_json",
        "reason",
        "checks_json",
        "status",
        "decision_reason",
        "claimed_at",
        "processed_at",
        "order_id",
    }


def test_alerts_link_exact_strategy_score(tmp_db: Engine) -> None:
    inspector = inspect(tmp_db)
    columns = {column["name"] for column in inspector.get_columns("alerts")}

    assert "strategy_score_id" in columns
    foreign_keys = inspector.get_foreign_keys("alerts")
    assert any(
        fk["constrained_columns"] == ["strategy_score_id"]
        and fk["referred_table"] == "strategy_scores"
        for fk in foreign_keys
    )


def _seed_sent_alert(engine: Engine) -> tuple[int, int, int]:
    now = datetime.now(UTC)
    with engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=now, spot=600.0)
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    rationale="test",
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        alert_id = int(
            conn.execute(
                insert(alerts).values(
                    strategy_score_id=score_id,
                    ts=now,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    score=80.0,
                    status="sent",
                    sent_ts=now,
                    telegram_msg_id=12345,
                )
            ).inserted_primary_key[0]
        )
    return snapshot_id, score_id, alert_id


def test_entry_review_insert_requires_exact_sent_alert(tmp_db: Engine) -> None:
    _, score_id, _ = _seed_sent_alert(tmp_db)

    with pytest.raises(IntegrityError, match="exact sent alert"):
        with tmp_db.begin() as conn:
            conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=score_id,
                    alert_id=None,
                    reviewed_at=datetime.now(UTC),
                    verdict="vetted_paper_candidate",
                    confidence=0.9,
                    sources_json=["a", "b"],
                    reason="invalid direct insertion",
                    checks_json={},
                    status="requested",
                )
            )


def test_entry_review_requires_proven_delivery_identity(tmp_db: Engine) -> None:
    _, score_id, alert_id = _seed_sent_alert(tmp_db)
    with tmp_db.begin() as conn:
        conn.execute(
            update(alerts)
            .where(alerts.c.id == alert_id)
            .values(telegram_msg_id=None)
        )

    with pytest.raises(IntegrityError, match="exact sent alert"):
        with tmp_db.begin() as conn:
            conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=score_id,
                    alert_id=alert_id,
                    reviewed_at=datetime.now(UTC),
                    verdict="vetted_paper_candidate",
                    confidence=0.9,
                    sources_json=["a", "b"],
                    reason="delivery identity missing",
                    checks_json={},
                    status="requested",
                )
            )


def test_entry_review_evidence_is_immutable(tmp_db: Engine) -> None:
    snapshot_id, score_id, alert_id = _seed_sent_alert(tmp_db)
    with tmp_db.begin() as conn:
        review_id = int(
            conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=score_id,
                    alert_id=alert_id,
                    reviewed_at=datetime.now(UTC),
                    verdict="vetted_paper_candidate",
                    confidence=0.9,
                    sources_json=["a", "b"],
                    reason="valid immutable evidence",
                    checks_json={"all": True},
                    status="requested",
                )
            ).inserted_primary_key[0]
        )

    with pytest.raises(IntegrityError, match="evidence is immutable"):
        with tmp_db.begin() as conn:
            conn.execute(
                update(entry_reviews)
                .where(entry_reviews.c.id == review_id)
                .values(confidence=0.1)
            )

    with pytest.raises(IntegrityError, match="reviewed strategy score is immutable"):
        with tmp_db.begin() as conn:
            conn.execute(
                update(strategy_scores)
                .where(strategy_scores.c.id == score_id)
                .values(score=99.0)
            )

    for table, row_id, message in (
        (entry_reviews, review_id, "audit rows cannot be deleted"),
        (alerts, alert_id, "reviewed alert cannot be deleted"),
        (strategy_scores, score_id, "reviewed strategy score cannot be deleted"),
        (snapshots, snapshot_id, "reviewed snapshot cannot be deleted"),
    ):
        with pytest.raises(IntegrityError, match=message):
            with tmp_db.begin() as conn:
                conn.execute(delete(table).where(table.c.id == row_id))


def test_entry_intent_consumption_is_immutable(tmp_db: Engine) -> None:
    _, score_id, _ = _seed_sent_alert(tmp_db)
    now = datetime.now(UTC)
    with tmp_db.begin() as conn:
        order_id = int(
            conn.execute(
                insert(orders).values(
                    strategy_score_id=score_id,
                    intent="open",
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=[],
                    quantity=1,
                    status="rejected",
                    staged_ts=now,
                    terminal_ts=now,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(entry_intent_consumptions).values(
                strategy_score_id=score_id,
                first_order_id=order_id,
                consumed_at=now,
            )
        )

    with pytest.raises(IntegrityError, match="consumption is immutable"):
        with tmp_db.begin() as conn:
            conn.execute(
                update(entry_intent_consumptions)
                .where(entry_intent_consumptions.c.strategy_score_id == score_id)
                .values(first_order_id=order_id + 1)
            )
    with pytest.raises(IntegrityError, match="consumption cannot be deleted"):
        with tmp_db.begin() as conn:
            conn.execute(
                delete(entry_intent_consumptions).where(
                    entry_intent_consumptions.c.strategy_score_id == score_id
                )
            )


def test_upgrade_reconciles_duplicate_terminal_legacy_entry_intents(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-duplicate-orders.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0014")

    engine = create_engine_for_path(db_path)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=now, spot=600.0)
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        for status in ("failed", "sent"):
            conn.execute(
                insert(alerts).values(
                    strategy_score_id=score_id,
                    ts=now,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    score=80.0,
                    status=status,
                    sent_ts=now if status == "sent" else None,
                    telegram_msg_id=12345 if status == "sent" else None,
                )
            )
        for status in ("skipped", "rejected"):
            conn.execute(
                insert(orders).values(
                    strategy_score_id=score_id,
                    intent="open",
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=[],
                    quantity=1,
                    status=status,
                    staged_ts=now,
                    terminal_ts=now,
                    reprice_count=0,
                )
            )
    engine.dispose()

    command.upgrade(cfg, "0015")
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(orders)).scalar_one() == 2
        assert conn.execute(select(func.count()).select_from(alerts)).scalar_one() == 2
        receipt = conn.execute(select(entry_intent_consumptions)).one()
        assert receipt.strategy_score_id == score_id
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    engine.dispose()

    command.downgrade(cfg, "0014")
    downgraded = create_engine_for_path(db_path)
    assert "entry_intent_consumptions" not in inspect(downgraded).get_table_names()
    with downgraded.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    downgraded.dispose()


def test_upgrade_resumes_after_correct_unique_score_index_prefix(
    tmp_path: Path,
) -> None:
    """A SQLite DDL interruption after the first 0015 statement is healable."""
    db_path = tmp_path / "partial-score-index.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0014")

    engine = create_engine_for_path(db_path)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX uq_strategy_scores_snapshot_strategy "
            "ON strategy_scores (snapshot_id, strategy)"
        )
    engine.dispose()

    command.upgrade(cfg, "head")

    upgraded = create_engine_for_path(db_path)
    inspector = inspect(upgraded)
    assert "entry_intent_consumptions" in inspector.get_table_names()
    assert "alert_id" in {
        column["name"] for column in inspector.get_columns("entry_reviews")
    }
    with upgraded.connect() as conn:
        assert conn.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == "0019"
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    upgraded.dispose()


def test_upgrade_rejects_same_name_partial_unique_score_index(
    tmp_path: Path,
) -> None:
    """A partial unique index cannot prove global score identity."""
    db_path = tmp_path / "partial-unique-score-index.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0014")

    engine = create_engine_for_path(db_path)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX uq_strategy_scores_snapshot_strategy "
            "ON strategy_scores (snapshot_id, strategy) "
            "WHERE strategy IS NULL"
        )
    engine.dispose()

    with pytest.raises(RuntimeError, match="incompatible strategy score identity index"):
        command.upgrade(cfg, "head")

    rejected = create_engine_for_path(db_path)
    inspector = inspect(rejected)
    assert "entry_intent_consumptions" not in inspector.get_table_names()
    assert "alert_id" not in {
        column["name"] for column in inspector.get_columns("entry_reviews")
    }
    with rejected.connect() as conn:
        assert conn.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == "0014"
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    rejected.dispose()


def test_upgrade_preflights_ambiguous_scores_before_any_sqlite_ddl(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ambiguous-scores.db"
    cfg = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "0014")

    engine = create_engine_for_path(db_path)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=now, spot=600.0)
            ).inserted_primary_key[0]
        )
        for score in (70.0, 71.0):
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="iron_condor",
                    score=score,
                    legs_json=[],
                    suggestion_json={},
                )
            )
    engine.dispose()

    with pytest.raises(RuntimeError, match="one strategy_scores row"):
        command.upgrade(cfg, "0015")

    unchanged = create_engine_for_path(db_path)
    inspector = inspect(unchanged)
    assert "entry_intent_consumptions" not in inspector.get_table_names()
    assert "alert_id" not in {
        column["name"] for column in inspector.get_columns("entry_reviews")
    }
    assert "uq_strategy_scores_snapshot_strategy" not in {
        index["name"] for index in inspector.get_indexes("strategy_scores")
    }
    with unchanged.connect() as conn:
        revision = conn.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
        assert revision == "0014"
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    unchanged.dispose()
