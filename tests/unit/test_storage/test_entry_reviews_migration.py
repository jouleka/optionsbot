"""Migration contract for Hermes entry-review requests."""

from __future__ import annotations

from sqlalchemy import Engine, inspect


def test_entry_reviews_table_is_migrated(tmp_db: Engine) -> None:
    inspector = inspect(tmp_db)

    assert "entry_reviews" in inspector.get_table_names()
    columns = {column["name"] for column in inspector.get_columns("entry_reviews")}
    assert columns == {
        "id",
        "strategy_score_id",
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
