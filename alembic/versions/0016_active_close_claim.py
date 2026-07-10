"""Enforce one active close order per entry.

Revision ID: 0016
Revises: 0015
Create Date: 2026-07-10 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | Sequence[str] | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTIVE_CLOSE = (
    "closes_order_id IS NOT NULL "
    "AND status IN ('staged', 'submitting', 'submitted', 'partial')"
)


def upgrade() -> None:
    op.create_index(
        "uq_orders_active_close_per_entry",
        "orders",
        ["closes_order_id"],
        unique=True,
        sqlite_where=sa.text(_ACTIVE_CLOSE),
        postgresql_where=sa.text(_ACTIVE_CLOSE),
    )


def downgrade() -> None:
    op.drop_index("uq_orders_active_close_per_entry", table_name="orders")
