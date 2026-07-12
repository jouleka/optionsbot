"""Enforce one ledger owner per broker order ID.

Revision ID: 0017
Revises: 0016
Create Date: 2026-07-11 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0017"
down_revision: str | Sequence[str] | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NON_NULL_BROKER_ID = "ib_order_id IS NOT NULL"


def upgrade() -> None:
    bind = op.get_bind()
    duplicate = bind.execute(
        sa.text(
            """
            SELECT ib_order_id
            FROM orders
            WHERE ib_order_id IS NOT NULL
            GROUP BY ib_order_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if duplicate is not None:
        # Ownership is ambiguous: do not guess which ledger row owns the broker
        # order. Quarantine every conflicting binding, retain the raw ID in the
        # audit message, and require human broker/ledger reconciliation.
        bind.execute(
            sa.text(
                """
                UPDATE orders
                SET last_error =
                        'migration 0017 quarantined duplicate broker order id ' ||
                        CAST(ib_order_id AS TEXT) ||
                        '; reconcile broker state before re-arming',
                    ib_order_id = NULL
                WHERE ib_order_id IN (
                    SELECT duplicate_id
                    FROM (
                        SELECT ib_order_id AS duplicate_id
                        FROM orders
                        WHERE ib_order_id IS NOT NULL
                        GROUP BY ib_order_id
                        HAVING COUNT(*) > 1
                    )
                )
                """
            )
        )
        reason = (
            "migration 0017 found ambiguous broker order ownership; "
            "reconcile broker and ledger state before re-arming"
        )
        updated = bind.execute(
            sa.text(
                """
                UPDATE execution_state
                SET killed = 1, reason = :reason, ts = CURRENT_TIMESTAMP
                WHERE id = 1
                """
            ),
            {"reason": reason},
        )
        if updated.rowcount != 1:
            bind.execute(
                sa.text(
                    """
                    INSERT INTO execution_state (id, killed, reason, ts)
                    VALUES (1, 1, :reason, CURRENT_TIMESTAMP)
                    """
                ),
                {"reason": reason},
            )
    op.create_index(
        "uq_orders_ib_order_id",
        "orders",
        ["ib_order_id"],
        unique=True,
        sqlite_where=sa.text(_NON_NULL_BROKER_ID),
        postgresql_where=sa.text(_NON_NULL_BROKER_ID),
    )


def downgrade() -> None:
    op.drop_index("uq_orders_ib_order_id", table_name="orders")
