"""pick_outcomes forward outcome ledger

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-03 00:00:00.000000

Adds the pick_outcomes table (IBK-99 Phase B): one realized-outcome row per
evaluated strategy_scores pick (terminal underlying at expiry -> realized_pnl /
win), for the forward validation ledger. UNIQUE(strategy_score_id) dedups.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'pick_outcomes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_score_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.Text(), nullable=False),
        sa.Column('strategy', sa.Text(), nullable=False),
        sa.Column('expiry', sa.Text(), nullable=False),
        sa.Column('entry_spot', sa.Float(), nullable=False),
        sa.Column('predicted_prob_profit', sa.Float(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('credit_or_debit', sa.Float(), nullable=True),
        sa.Column('max_profit', sa.Float(), nullable=True),
        sa.Column('max_loss', sa.Float(), nullable=True),
        sa.Column('risk_tier', sa.Text(), nullable=True),
        sa.Column('terminal_spot', sa.Float(), nullable=False),
        sa.Column('realized_pnl', sa.Float(), nullable=False),
        sa.Column('win', sa.Integer(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['strategy_score_id'], ['strategy_scores.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('strategy_score_id'),
    )


def downgrade() -> None:
    op.drop_table('pick_outcomes')
