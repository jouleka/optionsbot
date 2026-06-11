"""orders.closes_order_id linkage

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-11 00:00:00.000000

Adds orders.closes_order_id (IBK-129): a closing order points at the entry
it closes, so exit dedupe and realized-P&L pairing never infer.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0010'
down_revision: Union[str, Sequence[str], None] = '0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite: ALTER TABLE ADD COLUMN works; the FK is declared in the model
    # (SQLite doesn't enforce FKs added via ALTER, which is acceptable here).
    op.add_column('orders', sa.Column('closes_order_id', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('orders', 'closes_order_id')
