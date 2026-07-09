"""Hermes request_exit audit queue

Revision ID: 0013
Revises: 0012
Create Date: 2026-07-09 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0013'
down_revision: Union[str, Sequence[str], None] = '0012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'exit_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.Integer(), nullable=False),
        sa.Column('requested_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('catalyst_type', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('sources_json', sa.JSON(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), server_default='requested', nullable=False),
        sa.Column('decision_reason', sa.Text(), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('close_order_id', sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "status IN ('requested','refused','submitted','failed')",
            name='ck_exit_requests_status',
        ),
        sa.ForeignKeyConstraint(['position_id'], ['orders.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['close_order_id'], ['orders.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_exit_requests_status_requested_at', 'exit_requests', ['status', 'requested_at'])
    op.create_index('ix_exit_requests_position_requested_at', 'exit_requests', ['position_id', 'requested_at'])


def downgrade() -> None:
    op.drop_index('ix_exit_requests_position_requested_at', table_name='exit_requests')
    op.drop_index('ix_exit_requests_status_requested_at', table_name='exit_requests')
    op.drop_table('exit_requests')
