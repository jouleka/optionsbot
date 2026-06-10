"""orders + fills ledger

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-10 00:00:00.000000

Adds the order ledger (IBK-124): orders (one row per order intent, staged
before any network call; order_ref maps broker orders back to rows) and
fills (per-leg executions, UNIQUE execId for reconnect-replay dedupe).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0008'
down_revision: Union[str, Sequence[str], None] = '0007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'orders',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('strategy_score_id', sa.Integer(), nullable=True),
        sa.Column('intent', sa.Text(), nullable=False),
        sa.Column('symbol', sa.Text(), nullable=False),
        sa.Column('strategy', sa.Text(), nullable=False),
        sa.Column('legs_json', sa.JSON(), nullable=True),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('limit_price', sa.Float(), nullable=True),
        sa.Column('ib_order_id', sa.Integer(), nullable=True),
        sa.Column('ib_perm_id', sa.Integer(), nullable=True),
        sa.Column('order_ref', sa.Text(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('staged_ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('submitted_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('terminal_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('reprice_count', sa.Integer(), server_default='0', nullable=False),
        sa.CheckConstraint("intent IN ('open','close')", name='ck_orders_intent'),
        sa.CheckConstraint(
            "status IN ('staged','submitting','submitted','partial','filled',"
            "'cancelled','rejected','abandoned','skipped')",
            name='ck_orders_status',
        ),
        sa.ForeignKeyConstraint(
            ['strategy_score_id'], ['strategy_scores.id'], ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_ref'),
    )
    op.create_index('ix_orders_strategy_score_id', 'orders', ['strategy_score_id'])
    op.create_index('ix_orders_symbol', 'orders', ['symbol'])
    op.create_index('ix_orders_status', 'orders', ['status'])

    op.create_table(
        'fills',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.Integer(), nullable=False),
        sa.Column('ib_exec_id', sa.Text(), nullable=False),
        sa.Column('side', sa.Text(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('commission', sa.Float(), nullable=True),
        sa.Column('leg_con_id', sa.Integer(), nullable=True),
        sa.CheckConstraint("side IN ('BUY','SELL')", name='ck_fills_side'),
        sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('ib_exec_id'),
    )
    op.create_index('ix_fills_order_id', 'fills', ['order_id'])


def downgrade() -> None:
    op.drop_index('ix_fills_order_id', table_name='fills')
    op.drop_table('fills')
    op.drop_index('ix_orders_status', table_name='orders')
    op.drop_index('ix_orders_symbol', table_name='orders')
    op.drop_index('ix_orders_strategy_score_id', table_name='orders')
    op.drop_table('orders')
