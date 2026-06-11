"""order_quotes decision journal

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-11 00:00:00.000000

Adds order_quotes (IBK-127): the decision-quote journal — per-leg NBBO at
decision time and at every price-walk step, so realized fills can be scored
against the quotes the bot acted on (implementation shortfall).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0009'
down_revision: Union[str, Sequence[str], None] = '0008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'order_quotes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.Integer(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('step', sa.Integer(), server_default='0', nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('combo_bid', sa.Float(), nullable=True),
        sa.Column('combo_ask', sa.Float(), nullable=True),
        sa.Column('combo_mid', sa.Float(), nullable=True),
        sa.Column('target_net', sa.Float(), nullable=True),
        sa.Column('limit_price', sa.Float(), nullable=True),
        sa.Column('legs_json', sa.JSON(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('decision','step','final')", name='ck_order_quotes_kind'
        ),
        sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_order_quotes_order_id', 'order_quotes', ['order_id'])


def downgrade() -> None:
    op.drop_index('ix_order_quotes_order_id', table_name='order_quotes')
    op.drop_table('order_quotes')
