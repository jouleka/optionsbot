"""execution_state kill switch

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-10 00:00:00.000000

Adds execution_state (IBK-123): a singleton row (id=1) holding the execution
kill switch, persisted so a tripped kill survives daemon restarts and is only
cleared by an explicit /arm.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'execution_state',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('killed', sa.Integer(), server_default='0', nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('execution_state')
