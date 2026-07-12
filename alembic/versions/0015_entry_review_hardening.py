"""Fail-closed entry-review identity and at-most-once entry intents.

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-10 20:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0015"
down_revision: str | Sequence[str] | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VALIDATE_TRIGGER = "trg_entry_reviews_validate_insert"
_IMMUTABLE_TRIGGER = "trg_entry_reviews_evidence_immutable"
_SCORE_IMMUTABLE_TRIGGER = "trg_reviewed_strategy_score_immutable"
_ALERT_IMMUTABLE_TRIGGER = "trg_reviewed_alert_immutable"
_SNAPSHOT_IMMUTABLE_TRIGGER = "trg_reviewed_snapshot_immutable"
_REVIEW_DELETE_TRIGGER = "trg_entry_reviews_no_delete"
_SCORE_DELETE_TRIGGER = "trg_reviewed_strategy_score_no_delete"
_ALERT_DELETE_TRIGGER = "trg_reviewed_alert_no_delete"
_SNAPSHOT_DELETE_TRIGGER = "trg_reviewed_snapshot_no_delete"
_CONSUMPTION_UPDATE_TRIGGER = "trg_entry_consumption_no_update"
_CONSUMPTION_DELETE_TRIGGER = "trg_entry_consumption_no_delete"


def upgrade() -> None:
    # SQLite keeps successful DDL even when a later migration statement fails.
    # Preflight irreconcilable score ambiguity before creating any index/table,
    # so an old corrupt database is rejected without being partially migrated.
    duplicate_score = op.get_bind().execute(
        sa.text(
            """
            SELECT snapshot_id, strategy
              FROM strategy_scores
             GROUP BY snapshot_id, strategy
            HAVING COUNT(*) > 1
             LIMIT 1
            """
        )
    ).first()
    if duplicate_score is not None:
        raise RuntimeError(
            "0015 requires one strategy_scores row per snapshot and strategy"
        )

    score_indexes = {
        index["name"]: index
        for index in sa.inspect(op.get_bind()).get_indexes("strategy_scores")
    }
    score_identity_index = score_indexes.get(
        "uq_strategy_scores_snapshot_strategy"
    )
    if score_identity_index is None:
        op.create_index(
            "uq_strategy_scores_snapshot_strategy",
            "strategy_scores",
            ["snapshot_id", "strategy"],
            unique=True,
        )
    elif not (
        score_identity_index.get("unique")
        and score_identity_index.get("column_names")
        == ["snapshot_id", "strategy"]
    ):
        raise RuntimeError(
            "0015 found an incompatible strategy score identity index"
        )
    with op.batch_alter_table("entry_reviews") as batch_op:
        batch_op.add_column(sa.Column("alert_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_entry_reviews_alert_id",
            "alerts",
            ["alert_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_index("ix_entry_reviews_alert_id", ["alert_id"], unique=True)

    # Only an unambiguous historical alert may be attached automatically.
    # Ambiguous/unmatched legacy reviews remain auditable but are terminalized;
    # the daemon also refuses every review without an exact alert identity.
    op.execute(
        sa.text(
            """
            UPDATE entry_reviews
               SET alert_id = (
                   SELECT MIN(alerts.id)
                     FROM alerts
                    WHERE alerts.strategy_score_id = entry_reviews.strategy_score_id
               )
             WHERE (
                   SELECT COUNT(*)
                     FROM alerts
                    WHERE alerts.strategy_score_id = entry_reviews.strategy_score_id
               ) = 1
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE entry_reviews
               SET status = 'held',
                   decision_reason = 'migration held review without one exact alert identity',
                   claimed_at = NULL,
                   processed_at = COALESCE(processed_at, reviewed_at)
             WHERE alert_id IS NULL
               AND status IN ('requested', 'processing')
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE entry_reviews
               SET status = 'held',
                   decision_reason = 'migration held review without proven alert delivery',
                   claimed_at = NULL,
                   processed_at = COALESCE(processed_at, reviewed_at)
             WHERE alert_id IS NOT NULL
               AND status IN ('requested', 'processing')
               AND NOT EXISTS (
                   SELECT 1 FROM alerts
                    WHERE alerts.id = entry_reviews.alert_id
                      AND alerts.status = 'sent'
                      AND alerts.sent_ts IS NOT NULL
                      AND alerts.telegram_msg_id IS NOT NULL
                      AND alerts.sent_ts >= alerts.ts
                      AND entry_reviews.reviewed_at >= alerts.sent_ts
               )
            """
        )
    )

    # Permanent at-most-once admission receipts are separate from mutable order
    # rows. This safely reconciles valid 0014 histories that contain repeated
    # terminal attempts for one score instead of failing a partial SQLite DDL
    # migration on a new unique orders index.
    op.create_table(
        "entry_intent_consumptions",
        sa.Column("strategy_score_id", sa.Integer(), nullable=False),
        sa.Column("first_order_id", sa.Integer(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["strategy_score_id"], ["strategy_scores.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["first_order_id"], ["orders.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("strategy_score_id"),
        sa.UniqueConstraint("first_order_id"),
    )
    op.execute(
        sa.text(
            """
            INSERT INTO entry_intent_consumptions
                (strategy_score_id, first_order_id, consumed_at)
            SELECT strategy_score_id, MIN(id), MIN(staged_ts)
              FROM orders
             WHERE intent = 'open'
               AND strategy_score_id IS NOT NULL
             GROUP BY strategy_score_id
            """
        )
    )

    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_VALIDATE_TRIGGER}
            BEFORE INSERT ON entry_reviews
            WHEN NEW.alert_id IS NULL
              OR NOT EXISTS (
                  SELECT 1
                    FROM alerts
                   WHERE alerts.id = NEW.alert_id
                     AND alerts.strategy_score_id = NEW.strategy_score_id
                     AND alerts.status = 'sent'
                     AND alerts.sent_ts IS NOT NULL
                     AND alerts.telegram_msg_id IS NOT NULL
                     AND alerts.sent_ts >= alerts.ts
                     AND NEW.reviewed_at >= alerts.sent_ts
              )
            BEGIN
                SELECT RAISE(ABORT, 'entry review requires exact sent alert');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_IMMUTABLE_TRIGGER}
            BEFORE UPDATE OF
                strategy_score_id, alert_id, reviewed_at, verdict, confidence,
                sources_json, reason, checks_json
            ON entry_reviews
            BEGIN
                SELECT RAISE(ABORT, 'entry review evidence is immutable');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_SCORE_IMMUTABLE_TRIGGER}
            BEFORE UPDATE ON strategy_scores
            WHEN EXISTS (
                SELECT 1 FROM entry_reviews
                 WHERE entry_reviews.strategy_score_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'reviewed strategy score is immutable');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_ALERT_IMMUTABLE_TRIGGER}
            BEFORE UPDATE ON alerts
            WHEN EXISTS (
                SELECT 1 FROM entry_reviews
                 WHERE entry_reviews.alert_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'reviewed alert is immutable');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_SNAPSHOT_IMMUTABLE_TRIGGER}
            BEFORE UPDATE ON snapshots
            WHEN EXISTS (
                SELECT 1
                  FROM strategy_scores
                  JOIN entry_reviews
                    ON entry_reviews.strategy_score_id = strategy_scores.id
                 WHERE strategy_scores.snapshot_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'reviewed snapshot is immutable');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_REVIEW_DELETE_TRIGGER}
            BEFORE DELETE ON entry_reviews
            BEGIN
                SELECT RAISE(ABORT, 'entry review audit rows cannot be deleted');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_SCORE_DELETE_TRIGGER}
            BEFORE DELETE ON strategy_scores
            WHEN EXISTS (
                SELECT 1 FROM entry_reviews
                 WHERE entry_reviews.strategy_score_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'reviewed strategy score cannot be deleted');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_ALERT_DELETE_TRIGGER}
            BEFORE DELETE ON alerts
            WHEN EXISTS (
                SELECT 1 FROM entry_reviews
                 WHERE entry_reviews.alert_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'reviewed alert cannot be deleted');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_SNAPSHOT_DELETE_TRIGGER}
            BEFORE DELETE ON snapshots
            WHEN EXISTS (
                SELECT 1
                  FROM strategy_scores
                  JOIN entry_reviews
                    ON entry_reviews.strategy_score_id = strategy_scores.id
                 WHERE strategy_scores.snapshot_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'reviewed snapshot cannot be deleted');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_CONSUMPTION_UPDATE_TRIGGER}
            BEFORE UPDATE ON entry_intent_consumptions
            BEGIN
                SELECT RAISE(ABORT, 'entry intent consumption is immutable');
            END
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TRIGGER {_CONSUMPTION_DELETE_TRIGGER}
            BEFORE DELETE ON entry_intent_consumptions
            BEGIN
                SELECT RAISE(ABORT, 'entry intent consumption cannot be deleted');
            END
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_CONSUMPTION_DELETE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_CONSUMPTION_UPDATE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_SNAPSHOT_DELETE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_ALERT_DELETE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_SCORE_DELETE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_REVIEW_DELETE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_SNAPSHOT_IMMUTABLE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_ALERT_IMMUTABLE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_SCORE_IMMUTABLE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_IMMUTABLE_TRIGGER}"))
    op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_VALIDATE_TRIGGER}"))
    op.drop_table("entry_intent_consumptions")
    with op.batch_alter_table("entry_reviews") as batch_op:
        batch_op.drop_index("ix_entry_reviews_alert_id")
        batch_op.drop_constraint("fk_entry_reviews_alert_id", type_="foreignkey")
        batch_op.drop_column("alert_id")
    op.drop_index(
        "uq_strategy_scores_snapshot_strategy", table_name="strategy_scores"
    )
