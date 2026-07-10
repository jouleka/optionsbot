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
    bind = op.get_bind()
    duplicate = bind.execute(
        sa.text(
            """
            SELECT closes_order_id
            FROM orders
            WHERE closes_order_id IS NOT NULL
              AND status IN ('staged', 'submitting', 'submitted', 'partial')
            GROUP BY closes_order_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if duplicate is not None:
        # Preserve every historical row but retain only the oldest active claim.
        # Later duplicate attempts become terminal evidence, and the persisted
        # kill switch forces broker/ledger reconciliation before any re-arm.
        bind.execute(
            sa.text(
                """
                UPDATE orders
                SET status = 'abandoned',
                    terminal_ts = CURRENT_TIMESTAMP,
                    last_error =
                        'migration 0016 quarantined duplicate active close; ' ||
                        'reconcile broker state before re-arming'
                WHERE id IN (
                    SELECT candidate.id
                    FROM orders AS candidate
                    JOIN (
                        SELECT closes_order_id, MIN(id) AS keep_id
                        FROM orders
                        WHERE closes_order_id IS NOT NULL
                          AND status IN ('staged', 'submitting', 'submitted', 'partial')
                        GROUP BY closes_order_id
                        HAVING COUNT(*) > 1
                    ) AS duplicates
                      ON duplicates.closes_order_id = candidate.closes_order_id
                    WHERE candidate.status IN ('staged', 'submitting', 'submitted', 'partial')
                      AND candidate.id <> duplicates.keep_id
                )
                """
            )
        )
        kill_reason = (
            "migration 0016 found duplicate active closes; "
            "reconcile broker and ledger state before re-arming"
        )
        updated = bind.execute(
            sa.text(
                """
                UPDATE execution_state
                SET killed = 1,
                    reason = :reason,
                    ts = CURRENT_TIMESTAMP
                WHERE id = 1
                """
            ),
            {"reason": kill_reason},
        )
        if updated.rowcount != 1:
            bind.execute(
                sa.text(
                    """
                    INSERT INTO execution_state (id, killed, reason, ts)
                    VALUES (1, 1, :reason, CURRENT_TIMESTAMP)
                    """
                ),
                {"reason": kill_reason},
            )
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
