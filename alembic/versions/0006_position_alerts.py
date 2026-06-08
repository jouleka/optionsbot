"""position_alerts management-alert dedup

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-08 00:00:00.000000

Adds position_alerts (IBK-113): one row per management alert sent, keyed by
dedup_key, so should_manage_alert suppresses re-fires within the cooldown window
across daemon restarts.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0006'
down_revision: Union[str, Sequence[str], None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'position_alerts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('dedup_key', sa.Text(), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_position_alerts_dedup_key', 'position_alerts', ['dedup_key'])


def downgrade() -> None:
    op.drop_index('ix_position_alerts_dedup_key', table_name='position_alerts')
    op.drop_table('position_alerts')
