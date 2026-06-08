"""One position-management tick: evaluate open positions, dedup, alert (IBK-113).

Runs as a sibling of the scan tick. Reads the open book + underlying spots, turns
them into management alerts (analysis.management), and sends the ones that pass the
persisted cooldown (should_manage_alert). Never raises into the scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlalchemy import insert

from optionsbot.alerts.formatter import format_management_alert
from optionsbot.analysis.management import evaluate_position_triggers
from optionsbot.daemon.alert_dedup import should_manage_alert
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.ibkr.types import PortfolioPosition
from optionsbot.storage.schema import position_alerts

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManageRunSummary:
    positions_seen: int
    alerts_sent: int
    errors: list[str] = field(default_factory=list)


async def _fetch_spots(
    context: DaemonContext, positions: list[PortfolioPosition], errors: list[str]
) -> dict[str, float]:
    """Best-effort underlying spot per distinct short-option underlying. A failed
    fetch just omits that symbol (its assignment check is skipped)."""
    md = MarketDataClient(context.ibkr, context.resolver)
    symbols = {p.symbol for p in positions if p.sec_type == "OPT" and p.position < 0}
    spots: dict[str, float] = {}
    for sym in sorted(symbols):
        try:
            q = await md.get_stock_snapshot(sym)
        except Exception as e:  # noqa: BLE001 -- missing spot just skips assignment for sym
            log.exception("manage tick: spot fetch failed for %s", sym)
            errors.append(f"spot {sym}: {type(e).__name__}: {e}")
            continue
        spot = q.mid if q.mid is not None else q.last
        if spot is not None:
            spots[sym] = spot
    return spots


async def run_manage_tick(context: DaemonContext) -> ManageRunSummary:
    """Evaluate open positions and send deduped management alerts. No-op when the
    market is closed, management is disabled, or alerting is paused."""
    now = datetime.now(UTC)
    settings = context.settings
    if not is_market_open(now) or not settings.manage.enabled or context.alerting_paused:
        return ManageRunSummary(0, 0, [])
    errors: list[str] = []
    async with context.ibkr_lock:
        try:
            positions = await PositionsClient(context.ibkr).get_portfolio()
        except Exception as e:  # noqa: BLE001 -- whole-book read failure: bail this tick
            log.exception("manage tick: get_portfolio failed")
            return ManageRunSummary(0, 0, [f"get_portfolio: {type(e).__name__}: {e}"])
        spots = await _fetch_spots(context, positions, errors)
    alerts = evaluate_position_triggers(positions, spots, date.today(), settings.manage)
    sent = 0
    for alert in alerts:
        try:
            if not should_manage_alert(context.engine, settings, alert.dedup_key, now):
                continue
            await context.telegram.send_message(
                format_management_alert(alert), parse_mode=None
            )
            with context.engine.begin() as conn:
                conn.execute(
                    insert(position_alerts).values(dedup_key=alert.dedup_key, ts=now)
                )
            sent += 1
        except Exception as e:  # noqa: BLE001 -- one bad alert must not drop the rest
            log.exception("manage alert failed for %s", alert.dedup_key)
            errors.append(f"{alert.dedup_key}: {type(e).__name__}: {e}")
    return ManageRunSummary(len(positions), sent, errors)
