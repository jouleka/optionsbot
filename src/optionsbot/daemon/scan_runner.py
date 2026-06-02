"""One scan tick: market-hours gate + watchlist sweep + alert enqueue."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import insert, select

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.daemon.alert_pipeline import enqueue_alert, sweep_retries
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.observability import bind_log_context
from optionsbot.scan import scan_symbol
from optionsbot.scoring import DEFAULT_TOP_K, top_k
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
    scan_run_id = uuid.uuid4().hex
    # Per-symbol inner bind sets `symbol=...` and unbinds (not restores) on
    # exit, so an outer `symbol=None` would simply disappear after the first
    # iteration. Bind only scan_run_id at the outer scope.
    with bind_log_context(scan_run_id=scan_run_id):
        if not is_market_open(started_at):
            finished_at = datetime.now(UTC)
            _persist_scan_run(context, started_at, finished_at, 0, 0, [])
            log.info("scan tick skipped: market closed")
            return ScanRunSummary(
                started_at=started_at,
                finished_at=finished_at,
                tickers_scanned=0,
                alerts_enqueued=0,
                retries_dispatched=0,
                errors=[],
            )

        retries_dispatched = await sweep_retries(context)
        symbols = _load_watchlist(context)
        tickers_scanned = 0
        alerts_enqueued = 0
        errors: list[str] = []

        for sym, override in symbols:
            with bind_log_context(symbol=sym):
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
                selected = top_k(
                    result.scored,
                    k=DEFAULT_TOP_K,
                    threshold=context.settings.scan.score_threshold,
                )
                for scored in selected:
                    try:
                        # enqueue_alert returns True when a row was actually inserted,
                        # False when the dedup gate suppressed it. Increment only on
                        # True so scan_runs.alerts_fired matches reality.
                        if await enqueue_alert(context, sym, scored, result.snapshot_id):
                            alerts_enqueued += 1
                    except Exception as e:  # noqa: BLE001
                        log.exception("enqueue_alert failed for %s/%s", sym, scored.strategy_name)
                        errors.append(f"{sym}/{scored.strategy_name}: {type(e).__name__}: {e}")

        finished_at = datetime.now(UTC)
        _persist_scan_run(
            context, started_at, finished_at, tickers_scanned, alerts_enqueued, errors
        )
        return ScanRunSummary(
            started_at=started_at,
            finished_at=finished_at,
            tickers_scanned=tickers_scanned,
            alerts_enqueued=alerts_enqueued,
            retries_dispatched=retries_dispatched,
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
