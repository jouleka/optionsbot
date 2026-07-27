"""Add durable all-OTM expiration settlements.

Revision ID: 0020
Revises: 0019
Create Date: 2026-07-27 20:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0020"
down_revision: str | Sequence[str] | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "position_settlements",
        sa.Column("entry_order_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("expiry", sa.Text(), nullable=False),
        sa.Column("terminal_spot", sa.Float(), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=False),
        sa.Column("commissions", sa.Float(), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('expired_worthless')",
            name="ck_position_settlements_kind",
        ),
        sa.ForeignKeyConstraint(
            ["entry_order_id"],
            ["orders.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("entry_order_id"),
    )
    op.create_index(
        "ix_position_settlements_settled_at",
        "position_settlements",
        ["settled_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_position_settlements_settled_at",
        table_name="position_settlements",
    )
    op.drop_table("position_settlements")
