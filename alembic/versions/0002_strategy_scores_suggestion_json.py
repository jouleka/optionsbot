"""strategy_scores.suggestion_json

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-28 10:00:00.000000

Adds a JSON column to persist the StrategySuggestion fields (defined_risk,
credit_or_debit, max_loss, max_profit, prob_profit, suggested_quantity)
on each strategy_scores row. Without these, the daemon's retry path
(daemon/alert_pipeline._reconstruct_scored) had to fabricate
defined_risk=True and zero financials -- silently stripping the
"UNDEFINED RISK" warning and the credit/debit/max-loss figures from any
retry alert. With the column populated, retry alerts render exactly the
same payload as the first attempt.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'strategy_scores',
        sa.Column('suggestion_json', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('strategy_scores', 'suggestion_json')
