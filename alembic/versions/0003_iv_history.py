"""iv_history daily ATM-IV accumulation

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-01 00:00:00.000000

Adds the iv_history table: one ATM-IV value per (symbol, date), accumulated
forward by the scan loop so analysis.iv_rank has a trailing daily series
(IBKR provides no historical IV).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'iv_history',
        sa.Column('symbol', sa.Text(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('atm_iv', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('symbol', 'date'),
    )


def downgrade() -> None:
    op.drop_table('iv_history')
