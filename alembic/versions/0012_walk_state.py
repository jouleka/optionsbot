"""walk_state price-walk persistence

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-24 00:00:00.000000

Adds walk_state (Work-stream D1): one row per actively-walking order, written
on every price-walk step so a daemon restart can re-attach and resume the walk
(or issue one corrective reprice) instead of orphaning the in-memory asyncio
task until the TTL watcher cancels it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0012'
down_revision: Union[str, Sequence[str], None] = '0011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'walk_state',
        sa.Column('order_id', sa.Integer(), nullable=False),
        sa.Column('ib_order_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.Text(), nullable=False),
        sa.Column('legs_json', sa.JSON(), nullable=False),
        sa.Column('decision_mid', sa.Float(), nullable=False),
        sa.Column('budget', sa.Float(), nullable=False),
        sa.Column('increment', sa.Float(), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('prev_target', sa.Float(), nullable=False),
        sa.Column('updated_ts', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('order_id'),
    )


def downgrade() -> None:
    op.drop_table('walk_state')
