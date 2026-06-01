"""Daily ATM-IV history accumulation for IV-rank.

IBKR provides no historical IV, so the scan loop records one ATM-IV value per
symbol per day (latest scan of the day wins) and reads the trailing series
back to feed analysis.iv_rank.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from optionsbot.storage.schema import iv_history


def record_atm_iv(engine: Engine, symbol: str, day: date, atm_iv: float) -> None:
    """Upsert ``symbol``'s ATM IV for ``day`` (latest value wins per day)."""
    stmt = (
        sqlite_insert(iv_history)
        .values(symbol=symbol, date=day, atm_iv=atm_iv)
        .on_conflict_do_update(
            index_elements=["symbol", "date"],
            set_={"atm_iv": atm_iv},
        )
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def read_atm_iv_history(engine: Engine, symbol: str) -> pd.Series:
    """Return ``symbol``'s daily ATM IV as a float Series, oldest -> newest."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(iv_history.c.atm_iv)
            .where(iv_history.c.symbol == symbol)
            .order_by(iv_history.c.date)
        ).fetchall()
    return pd.Series([row.atm_iv for row in rows], dtype=float)
