"""Alert dedup predicate (IBK-62).

The dedup rule keeps the alert stream from spamming Claude on every tick
when a strategy stays in the top-K. We re-fire only when EITHER the
cooldown window has elapsed since the last 'sent' alert for the same
(symbol, strategy) pair, OR the score has jumped by more than the
configured delta (default 10 points). Strict greater-than on the delta:
"+10 exactly" is treated as noise; we want a genuine improvement.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, desc, select

from optionsbot.config import Settings
from optionsbot.storage.schema import alerts


def should_alert(
    engine: Engine,
    settings: Settings,
    symbol: str,
    strategy: str,
    score: float,
    now: datetime | None = None,
) -> bool:
    """Return True iff this (symbol, strategy, score) should be sent.

    Decision tree:
      * No prior 'sent' alert for the pair  -> True
      * Past cooldown window                 -> True
      * Within cooldown, score jumped > delta -> True
      * Otherwise                             -> False
    """
    now = now if now is not None else datetime.now(UTC)
    with engine.connect() as conn:
        row = conn.execute(
            select(alerts.c.ts, alerts.c.score)
            .where(alerts.c.symbol == symbol)
            .where(alerts.c.strategy == strategy)
            .where(alerts.c.status == "sent")
            .order_by(desc(alerts.c.ts))
            .limit(1)
        ).first()
    if row is None:
        return True
    last_ts: datetime = row.ts
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    cooldown = timedelta(hours=settings.scan.alert_cooldown_hours)
    if now - last_ts >= cooldown:
        return True
    last_score = float(row.score)
    return (score - last_score) > settings.scan.alert_rescore_delta
