"""Add the persistent Hermes overlay correctness breaker.

Revision ID: 0018
Revises: 0017
Create Date: 2026-07-15 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0018"
down_revision: str | Sequence[str] | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "hermes_overlay_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Integer(), server_default="1", nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("judgeable", sa.Integer(), server_default="0", nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.CheckConstraint("id = 1", name="ck_hermes_overlay_state_singleton"),
        sa.CheckConstraint("enabled IN (0, 1)", name="ck_hermes_overlay_state_enabled"),
        sa.CheckConstraint("judgeable >= 0", name="ck_hermes_overlay_state_judgeable"),
        sa.CheckConstraint(
            "accuracy IS NULL OR (accuracy >= 0.0 AND accuracy <= 1.0)",
            name="ck_hermes_overlay_state_accuracy",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.bulk_insert(
        sa.table(
            "hermes_overlay_state",
            sa.column("id", sa.Integer()),
            sa.column("enabled", sa.Integer()),
            sa.column("judgeable", sa.Integer()),
        ),
        [{"id": 1, "enabled": 1, "judgeable": 0}],
    )


def downgrade() -> None:
    op.drop_table("hermes_overlay_state")
