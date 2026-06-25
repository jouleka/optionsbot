"""execution_state day-start net-liq baseline

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-24 00:00:00.000000

Phase 0 work-stream B: persist the per-session day-start net liquidation (and
the NYSE session date it belongs to) on the singleton execution_state row so
the net-liq drawdown circuit breaker survives an intraday daemon restart.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0011'
down_revision: Union[str, Sequence[str], None] = '0010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('execution_state', sa.Column('day_start_net_liq', sa.Float(), nullable=True))
    op.add_column('execution_state', sa.Column('day_start_session', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('execution_state', 'day_start_session')
    op.drop_column('execution_state', 'day_start_net_liq')
