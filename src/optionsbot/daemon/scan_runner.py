"""One scan tick: market-hours gate + watchlist sweep + alert enqueue."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import insert, select

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.daemon.alert_pipeline import enqueue_alert
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.scan import scan_symbol
from optionsbot.scoring import DEFAULT_THRESHOLD, DEFAULT_TOP_K, top_k
from optionsbot.storage.schema import scan_runs, watchlist

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScanRunSummary:
    started_at: datetime
    finished_at: datetime
    tickers_scanned: int
    alerts_enqueued: int
    retries_dispatched: int
    errors: list[str] = field(default_factory=list)


async def run_scan_tick(context: DaemonContext) -> ScanRunSummary:
    """One scheduler tick: gate on market hours, scan the watchlist, enqueue alerts.

    Returns a ScanRunSummary that the scheduler can log / persist. Errors per
    symbol are collected (not raised) so a single failure doesn't poison the
    whole tick; a `scan_runs` row is always written even when the market is
    closed (with tickers_scanned=0) so the daemon's heartbeat is visible.
    """
    started_at = datetime.now(UTC)
    if not is_market_open(started_at):
        finished_at = datetime.now(UTC)
        _persist_scan_run(context, started_at, finished_at, 0, 0, [])
        return ScanRunSummary(
            started_at=started_at,
            finished_at=finished_at,
            tickers_scanned=0,
            alerts_enqueued=0,
            retries_dispatched=0,
            errors=[],
        )

    symbols = _load_watchlist(context)
    tickers_scanned = 0
    alerts_enqueued = 0
    errors: list[str] = []

    for sym, override in symbols:
        try:
            result = await scan_symbol(
                sym, context.ibkr, context.engine, context.settings,
                resolver=context.resolver,
                view_override=override,
            )
        except Exception as e:  # noqa: BLE001 -- per-symbol failures are heterogeneous
            log.exception("scan_symbol failed for %s", sym)
            errors.append(f"{sym}: {type(e).__name__}: {e}")
            continue
        tickers_scanned += 1
        selected = top_k(result.scored, k=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD)
        for scored in selected:
            try:
                await enqueue_alert(context, sym, scored, result.snapshot_id)
                alerts_enqueued += 1
            except Exception as e:  # noqa: BLE001
                log.exception("enqueue_alert failed for %s/%s", sym, scored.strategy_name)
                errors.append(f"{sym}/{scored.strategy_name}: {type(e).__name__}: {e}")

    finished_at = datetime.now(UTC)
    _persist_scan_run(context, started_at, finished_at, tickers_scanned, alerts_enqueued, errors)
    return ScanRunSummary(
        started_at=started_at,
        finished_at=finished_at,
        tickers_scanned=tickers_scanned,
        alerts_enqueued=alerts_enqueued,
        retries_dispatched=0,  # Task 5 wires sweep-retries into this number.
        # Defensive copy: frozen=True freezes the field binding but not the
        # list itself; future callers mutating summary.errors must not
        # silently mutate the local list that already shaped the persisted
        # scan_runs row.
        errors=list(errors),
    )


def _load_watchlist(
    context: DaemonContext,
) -> list[tuple[str, tuple[Direction | None, IVRegime | None] | None]]:
    """Return [(symbol, view_override_or_None), ...] from the watchlist table."""
    stmt = select(
        watchlist.c.symbol,
        watchlist.c.view_override_dir,
        watchlist.c.view_override_iv,
    )
    out: list[tuple[str, tuple[Direction | None, IVRegime | None] | None]] = []
    with context.engine.connect() as conn:
        for row in conn.execute(stmt).fetchall():
            override: tuple[Direction | None, IVRegime | None] | None
            if row.view_override_dir is None and row.view_override_iv is None:
                override = None
            else:
                override = (
                    cast("Direction | None", row.view_override_dir),
                    cast("IVRegime | None", row.view_override_iv),
                )
            out.append((row.symbol, override))
    return out


def _persist_scan_run(
    context: DaemonContext,
    started_at: datetime,
    finished_at: datetime,
    tickers_scanned: int,
    alerts_enqueued: int,
    errors: list[str],
) -> None:
    with context.engine.begin() as conn:
        conn.execute(
            insert(scan_runs).values(
                started=started_at,
                finished=finished_at,
                tickers_scanned=tickers_scanned,
                alerts_fired=alerts_enqueued,
                errors_json=errors if errors else None,
            )
        )
