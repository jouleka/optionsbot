"""Allow intrinsic-value expiration settlements.

Revision ID: 0022
Revises: 0021
Create Date: 2026-07-30 18:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0022"
down_revision: str | Sequence[str] | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("position_settlements") as batch:
        batch.drop_constraint(
            "ck_position_settlements_kind",
            type_="check",
        )
        batch.create_check_constraint(
            "ck_position_settlements_kind",
            "kind IN ('expired_worthless','expired_intrinsic')",
        )


def downgrade() -> None:
    with op.batch_alter_table("position_settlements") as batch:
        batch.drop_constraint(
            "ck_position_settlements_kind",
            type_="check",
        )
        batch.create_check_constraint(
            "ck_position_settlements_kind",
            "kind IN ('expired_worthless')",
        )
