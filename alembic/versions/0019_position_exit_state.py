"""Add durable per-position exit high-water marks.

Revision ID: 0019
Revises: 0018
Create Date: 2026-07-27 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0019"
down_revision: str | Sequence[str] | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "position_exit_state",
        sa.Column("entry_order_id", sa.Integer(), nullable=False),
        sa.Column("peak_pnl_per_unit", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["entry_order_id"],
            ["orders.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("entry_order_id"),
    )


def downgrade() -> None:
    op.drop_table("position_exit_state")
