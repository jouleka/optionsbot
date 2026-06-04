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
from optionsbot.ibkr.history import HistoryClient
from optionsbot.observability import bind_log_context
from optionsbot.scan import scan_symbol
from optionsbot.scoring import ScoredStrategy
from optionsbot.screener.screen import screen_universe
from optionsbot.screener.universe import DEFAULT_UNIVERSE
from optionsbot.storage.schema import scan_runs, watchlist

log = logging.getLogger(__name__)


def rank_alert_candidates(
    picks: list[tuple[str, ScoredStrategy, int]],
    score_floor: float,
) -> list[tuple[str, ScoredStrategy, int]]:
    """Picks scoring >= ``score_floor``, sorted by score descending (best first)."""
    above = [p for p in picks if p[1].score >= score_floor]
    above.sort(key=lambda p: p[1].score, reverse=True)
    return above


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
        async with context.ibkr_lock:
            symbols = await _resolve_scan_symbols(context)
            tickers_scanned = 0
            errors: list[str] = []
            all_picks: list[tuple[str, ScoredStrategy, int]] = []

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
                    for scored in result.scored:
                        all_picks.append((sym, scored, result.snapshot_id))

        # Alert the day's best: floor by score, rank desc, enqueue the top N that
        # pass dedup. Counting only successful (dedup-passed) enqueues means a
        # cooldown'd pick doesn't waste a slot. Across the whole tick, not per symbol.
        alerts_enqueued = 0
        for sym, scored, snap_id in rank_alert_candidates(
            all_picks, context.settings.scan.score_threshold
        ):
            if alerts_enqueued >= context.settings.scan.alert_top_n:
                break
            try:
                if await enqueue_alert(context, sym, scored, snap_id):
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


async def _resolve_scan_symbols(
    context: DaemonContext,
) -> list[tuple[str, tuple[Direction | None, IVRegime | None] | None]]:
    """Symbols to scan this tick: the watchlist, augmented (when
    ``scan.auto_screen`` is on) with the top screened universe candidates.

    Watchlist entries keep their view_override; screened-only names get
    override None; on overlap the watchlist override wins (so nothing the user
    configured is lost). If screening raises, log and fall back to the watchlist
    alone -- screening must never abort a tick.
    """
    watchlist_symbols = _load_watchlist(context)
    if not context.settings.scan.auto_screen:
        return watchlist_symbols

    try:
        history = HistoryClient(context.ibkr, context.resolver)
        universe = context.settings.screener.universe or DEFAULT_UNIVERSE
        candidates = await screen_universe(
            history, universe, context.settings.screener.min_dollar_volume
        )
    except Exception:  # noqa: BLE001 -- screening must never abort the tick
        log.exception("auto-screen failed; falling back to watchlist only")
        return watchlist_symbols

    seen = {sym for sym, _ in watchlist_symbols}
    resolved = list(watchlist_symbols)
    for cand in candidates[: context.settings.screener.scan_top_n]:
        if cand.symbol not in seen:
            resolved.append((cand.symbol, None))
            seen.add(cand.symbol)
    return resolved


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
