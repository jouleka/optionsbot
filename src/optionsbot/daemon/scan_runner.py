"""One scan tick: market-hours gate + watchlist sweep + alert enqueue."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import insert, select

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.daemon.alert_pipeline import enqueue_alert, sweep_retries
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.gateway_health import BUDGET_TIMEOUT_SUFFIX
from optionsbot.daemon.market_hours import is_market_open, nyse_session_date
from optionsbot.ibkr.history import HistoryClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.observability import bind_log_context
from optionsbot.scan import scan_symbol
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.composite import edge_sort_key, has_positive_edge
from optionsbot.screener.screen import screen_universe
from optionsbot.screener.universe import DEFAULT_UNIVERSE
from optionsbot.storage.schema import scan_runs, watchlist
from optionsbot.strategies.base import StrategySuggestion

log = logging.getLogger(__name__)


def is_proposable(
    sug: StrategySuggestion, account_value_usd: float | None, single_trade_cap_pct: float
) -> bool:
    """A pick worth surfacing at this bankroll: defined-risk AND its per-contract
    max_loss fits the single-trade cap (equity_usd * single_trade_cap_pct).
    Fail-closed: unknown equity or undefined risk -> not proposable. max_loss is
    USD; account_value_usd is net-liq already converted to USD (see
    AccountSummary.net_liquidation_usd)."""
    if not sug.defined_risk or sug.max_loss is None:
        return False
    if account_value_usd is None:
        return False
    return float(sug.max_loss) <= account_value_usd * single_trade_cap_pct


def candidate_admission_blockers(
    scored: ScoredStrategy,
    score_floor: float,
    account_value_usd: float | None,
    single_trade_cap_pct: float,
) -> list[str]:
    """Return stable, machine-readable reasons a candidate cannot be surfaced.

    Hermes-originated proposals use these same reasons after OptionsBot rebuilds
    the idea from live data.  Keeping the explanation beside the admission
    predicate prevents the learning loop from seeing a vague "one of several
    gates failed" result that it cannot adapt to on its next pass.
    """
    blockers: list[str] = []
    suggestion = scored.suggestion
    if scored.score < score_floor:
        blockers.append(
            f"score_below_floor(score={scored.score:.2f},floor={score_floor:.2f})"
        )
    if not has_positive_edge(suggestion):
        expected_value = suggestion.expected_value
        ev_text = (
            f"{float(expected_value):.2f}"
            if isinstance(expected_value, int | float) and math.isfinite(expected_value)
            else "unavailable"
        )
        blockers.append(f"non_positive_edge(expected_value={ev_text})")
    if not suggestion.defined_risk or suggestion.max_loss is None:
        blockers.append("undefined_or_missing_max_loss")
    elif account_value_usd is None:
        blockers.append("live_equity_unavailable")
    else:
        max_loss = float(suggestion.max_loss)
        risk_cap = float(account_value_usd) * float(single_trade_cap_pct)
        if max_loss > risk_cap:
            blockers.append(
                "single_contract_risk_over_cap("
                f"max_loss={max_loss:.2f},cap={risk_cap:.2f},"
                f"cap_pct={single_trade_cap_pct:.4f})"
            )
    return blockers


def rank_alert_candidates(
    picks: list[tuple[str, ScoredStrategy, int]],
    score_floor: float,
    account_value_usd: float | None = None,
    single_trade_cap_pct: float = 0.10,
) -> list[tuple[str, ScoredStrategy, int]]:
    """Alert-worthy picks: ``score >= score_floor``, positive edge (EV>0), AND
    proposable at the current bankroll; sorted by sign-aware edge descending.

    The positive-edge filter (IBK-106) means the daemon auto-alerts only genuine
    vol-premium edge; on a no-edge tick this returns ``[]`` and nothing is
    enqueued. On-demand /scan + CLI still SHOW no-edge picks (with a banner);
    only auto-alerting is suppressed.

    Affordability (IBK-134/IBK-122): a pick must be defined-risk AND its
    per-contract ``max_loss`` (USD) must fit within the single-trade cap
    (equity_usd * single_trade_cap_pct). Fail-closed: unknown equity or
    undefined risk -> not proposable, pick is dropped. max_loss is USD;
    account_value_usd is net-liq already converted to USD (see
    AccountSummary.net_liquidation_usd). The precise execution money-gates
    (whatIf margin vs available-funds) remain downstream in execute_pick.
    """
    above = [
        p
        for p in picks
        if not candidate_admission_blockers(
            p[1],
            score_floor,
            account_value_usd,
            single_trade_cap_pct,
        )
    ]
    above.sort(key=lambda p: edge_sort_key(p[1].suggestion), reverse=True)
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
        # IBK-148: bound the long-lived resolver cache -- drop option contracts
        # that expired on a prior trading day before they accumulate forever.
        # Runs before the market gate so cleanup happens on closed days too.
        evicted = context.resolver.prune_expired(nyse_session_date(started_at))
        if evicted:
            log.info("pruned %d expired contracts from resolver cache", evicted)
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
                        # IBK-149: bound each symbol so one wedged dependency
                        # (a hung IB Gateway option lookup) can't stall the whole
                        # tick and starve orders management. TimeoutError is
                        # caught below -> the symbol is skipped, the tick proceeds.
                        result = await asyncio.wait_for(
                            scan_symbol(
                                sym, context.ibkr, context.engine, context.settings,
                                resolver=context.resolver,
                                view_override=override,
                            ),
                            timeout=context.settings.scan.scan_symbol_timeout_s,
                        )
                    except TimeoutError:
                        log.warning(
                            "scan_symbol(%s) exceeded %.0fs budget; skipping",
                            sym,
                            context.settings.scan.scan_symbol_timeout_s,
                        )
                        # Suffix is the shared constant so gateway_health's
                        # wedge detection can never drift from this format.
                        errors.append(f"{sym}: {BUDGET_TIMEOUT_SUFFIX}")
                        continue
                    except Exception as e:  # noqa: BLE001 -- per-symbol failures are heterogeneous
                        log.exception("scan_symbol failed for %s", sym)
                        errors.append(f"{sym}: {type(e).__name__}: {e}")
                        continue
                    tickers_scanned += 1
                    for scored in result.scored:
                        all_picks.append((sym, scored, result.snapshot_id))

            # Live net-liq (USD), fetched once per tick under the same lock, so
            # the affordability filter can drop picks that exceed the single-trade
            # cap (e.g. a $36k CSP on a $5k account). Fail-closed on error:
            # account_value_usd stays None and rank_alert_candidates drops
            # everything that requires an affordability check.
            account_value_usd: float | None = None
            try:
                # IBK-149: bound this end-of-tick IBKR await too -- it runs under
                # ibkr_lock after the symbol loop, so a Gateway that wedges here
                # would hang the whole tick and starve orders management. A
                # timeout is caught below -> affordability filter simply off.
                _summary = await asyncio.wait_for(
                    PositionsClient(context.ibkr).get_account_summary(),
                    timeout=context.settings.scan.scan_symbol_timeout_s,
                )
                if _summary.net_liquidation_usd is not None:
                    account_value_usd = float(_summary.net_liquidation_usd)
            except Exception:  # noqa: BLE001 -- net-liq is advisory (incl. timeout); never abort a tick
                log.exception("net-liq fetch failed/timed out; affordability filter off this tick")

        # Alert the day's best: floor by score, rank desc, enqueue the top N that
        # pass dedup. Counting only successful (dedup-passed) enqueues means a
        # cooldown'd pick doesn't waste a slot. Across the whole tick, not per symbol.
        alerts_enqueued = 0
        alerted = []
        if not context.alerting_paused:
            candidates = rank_alert_candidates(
                all_picks,
                context.settings.scan.score_threshold,
                account_value_usd,
                context.settings.execution.max_single_trade_risk_pct,
            )
            if not candidates and any(
                scored.score >= context.settings.scan.score_threshold
                for _, scored, _ in all_picks
            ):
                log.info(
                    "no-edge tick: pick(s) passed the score floor but none had "
                    "positive edge; suppressing alerts"
                )
            for sym, scored, snap_id in candidates:
                if alerts_enqueued >= context.settings.scan.alert_top_n:
                    break
                try:
                    if await enqueue_alert(context, sym, scored, snap_id):
                        alerts_enqueued += 1
                        alerted.append((sym, scored, snap_id))
                except Exception as e:  # noqa: BLE001
                    log.exception("enqueue_alert failed for %s/%s", sym, scored.strategy_name)
                    errors.append(f"{sym}/{scored.strategy_name}: {type(e).__name__}: {e}")

        # IBK-130: full-auto entries — the alerted picks run the same
        # execute_pick pipeline as /execute (every gate still applies).
        if alerted and context.settings.execution.mode == "auto":
            try:
                from optionsbot.daemon.auto_executor import auto_execute_candidates

                await auto_execute_candidates(context, alerted)
            except Exception:  # noqa: BLE001 -- auto entries must not poison the scan
                log.exception("auto-execute pass failed")

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
        # IBK-149: bound the universe screen so a wedged IB Gateway during
        # screening (it runs BEFORE the symbol loop) can't hang the whole tick.
        # A timeout is caught below -> fall back to watchlist-only.
        candidates = await asyncio.wait_for(
            screen_universe(
                history, universe, context.settings.screener.min_dollar_volume
            ),
            timeout=context.settings.scan.screen_timeout_s,
        )
    except Exception:  # noqa: BLE001 -- screening (incl. timeout) must never abort the tick
        log.exception("auto-screen failed/timed out; falling back to watchlist only")
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
