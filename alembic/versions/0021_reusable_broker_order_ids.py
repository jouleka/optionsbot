"""Scope broker order-ID uniqueness to active orders.

Revision ID: 0021
Revises: 0020
Create Date: 2026-07-29 19:15:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0021"
down_revision: str | Sequence[str] | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTIVE_BROKER_ID = (
    "ib_order_id IS NOT NULL "
    "AND status IN ('staged', 'submitting', 'submitted', 'partial')"
)
_NON_NULL_BROKER_ID = "ib_order_id IS NOT NULL"


def upgrade() -> None:
    # orderId is allocated by TWS/Gateway per client session and can be reused
    # after its sequence resets.  Preserve terminal audit history while still
    # preventing two simultaneously actionable ledger rows from owning one ID.
    op.drop_index("uq_orders_ib_order_id", table_name="orders")
    op.create_index(
        "uq_orders_ib_order_id",
        "orders",
        ["ib_order_id"],
        unique=True,
        sqlite_where=sa.text(_ACTIVE_BROKER_ID),
        postgresql_where=sa.text(_ACTIVE_BROKER_ID),
    )


def downgrade() -> None:
    op.drop_index("uq_orders_ib_order_id", table_name="orders")
    op.create_index(
        "uq_orders_ib_order_id",
        "orders",
        ["ib_order_id"],
        unique=True,
        sqlite_where=sa.text(_NON_NULL_BROKER_ID),
        postgresql_where=sa.text(_NON_NULL_BROKER_ID),
    )
