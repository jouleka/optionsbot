"""symbol_news headline cache

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-07 00:00:00.000000

Adds the symbol_news table (IBK-108): one upserted row per symbol caching the
latest yfinance headlines + fetch time, so daily_brief reads news from cache
(instant) and scans refresh it at most every throttle_hours.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'symbol_news',
        sa.Column('symbol', sa.Text(), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('headlines_json', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('symbol'),
    )


def downgrade() -> None:
    op.drop_table('symbol_news')
