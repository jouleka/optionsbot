"""Hermes entry-review audit queue.

Revision ID: 0014
Revises: 0013
Create Date: 2026-07-10 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("alerts") as batch_op:
        batch_op.add_column(sa.Column("strategy_score_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_alerts_strategy_score_id",
            "strategy_scores",
            ["strategy_score_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_alerts_strategy_score_id", ["strategy_score_id"], unique=False
        )

    op.create_table(
        "entry_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_score_id", sa.Integer(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sources_json", sa.JSON(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("checks_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.Text(), server_default="requested", nullable=False),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("order_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "verdict IN ('vetted_paper_candidate','watch_only','no_trade')",
            name="ck_entry_reviews_verdict",
        ),
        sa.CheckConstraint(
            "status IN ('requested','processing','held','refused','submitted','expired','failed')",
            name="ck_entry_reviews_status",
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_entry_reviews_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_score_id"], ["strategy_scores.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("strategy_score_id", name="uq_entry_reviews_strategy_score_id"),
    )
    op.create_index(
        "ix_entry_reviews_status_reviewed_at",
        "entry_reviews",
        ["status", "reviewed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_entry_reviews_status_reviewed_at", table_name="entry_reviews")
    op.drop_table("entry_reviews")
    with op.batch_alter_table("alerts") as batch_op:
        batch_op.drop_index("ix_alerts_strategy_score_id")
        batch_op.drop_constraint("fk_alerts_strategy_score_id", type_="foreignkey")
        batch_op.drop_column("strategy_score_id")
