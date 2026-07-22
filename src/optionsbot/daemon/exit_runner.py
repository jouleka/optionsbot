"""Automated exit tick (IBK-129): bot positions → closing orders.

Runs as a sibling of the scan/manage ticks (15-minute cadence — right for
50%-profit / DTE rules; the 1-minute order watcher manages the resulting
working orders). Only ledger-attributed positions are ever touched: a
position is a FILLED open-intent order with no filled close. Manual
positions remain alert-only.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import (
    is_market_open,
    minutes_to_nyse_close,
    nyse_session_date,
    nyse_session_start_utc,
)
from optionsbot.execution.close_safety import (
    NonAtomicCloseError,
    assert_atomic_close_legs,
    find_naked_short_legs,
)
from optionsbot.execution.engine import combo_mid
from optionsbot.execution.exit_requests import (
    ExitRequestGateInput,
    HermesLossCapDecision,
    QuoteGateState,
    evaluate_exit_request_gate,
    evaluate_hermes_loss_cap,
)
from optionsbot.execution.exits import evaluate_exit
from optionsbot.execution.gate import can_execute, can_reduce_risk
from optionsbot.execution.orders import (
    FAILED_TERMINAL_STATUSES,
    CloseAlreadyClaimed,
    OrderRecord,
    RealizedPnLUnavailable,
    get_order,
    net_premium,
    open_close_for,
    realized_close_pairs,
    set_order_leg_contracts,
    set_order_note,
    stage_close_order,
    transition,
)
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.execution.walk import (
    combo_bid_ask,
    price_increment_for,
    round_to_increment,
    slippage_budget,
)
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.types import OptionQuote
from optionsbot.storage.schema import execution_state, exit_requests, fills, orders

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExitsTickSummary:
    positions: int
    closes_submitted: int
    errors: int


def _exec_md(context: DaemonContext) -> MarketDataClient | None:
    """Quote source for exit pricing: the EXEC connection (lock-free; the
    scan tick has just released ibkr_lock but /scan can grab it any time).
    Patchable seam for tests."""
    if context.exec_ibkr is None:
        return None
    return MarketDataClient(context.exec_ibkr, context.resolver)


def _open_entries(context: DaemonContext) -> list[OrderRecord]:
    engine = context.engine
    with engine.connect() as conn:
        entry_rows = conn.execute(
            select(orders.c.id).where(orders.c.intent == "open").where(orders.c.status == "filled")
        ).fetchall()
        closed_ids = {
            row.closes_order_id
            for row in conn.execute(
                select(orders.c.closes_order_id)
                .where(orders.c.intent == "close")
                .where(orders.c.status == "filled")
            ).fetchall()
        }
    records = []
    for row in entry_rows:
        if row.id in closed_ids:
            continue
        record = get_order(engine, row.id)
        if record is not None:
            records.append(record)
    return records


def _half_closed(engine: Engine, entry_id: int) -> bool:
    """True when a dead (non-filled terminal) close for this entry recorded
    any fills — the position is partially flat and must not be re-closed at
    full quantity (Opus IBK-129 critical)."""
    with engine.connect() as conn:
        closes = conn.execute(
            select(orders.c.id, orders.c.status).where(orders.c.closes_order_id == entry_id)
        ).fetchall()
        for row in closes:
            if row.status not in FAILED_TERMINAL_STATUSES:
                continue
            has_fill = conn.execute(
                select(fills.c.id).where(fills.c.order_id == row.id).limit(1)
            ).first()
            if has_fill is not None:
                return True
    return False


def _min_dte(legs: list[dict[str, object]], now: datetime) -> int | None:
    dtes = []
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT" or not leg.get("expiry"):
            continue
        expiry = str(leg["expiry"])
        try:
            exp_date = datetime(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]), tzinfo=UTC)
        except ValueError:
            continue
        dtes.append((exp_date.date() - now.date()).days)
    return min(dtes) if dtes else None


def _exit_quote_readiness(
    quotes: dict[tuple[str, float, str], OptionQuote],
    now: datetime,
    max_age_seconds: int,
) -> tuple[dict[tuple[str, float, str], float], str | None]:
    """Require exact live provenance and known, current timestamps for every quote."""
    ages: dict[tuple[str, float, str], float] = {}
    if not quotes:
        return ages, "quotes unavailable"
    for spec, quote in quotes.items():
        if quote.delayed is not False:
            return ages, "delayed or unknown quote delivery"
        if not isinstance(quote.ts, datetime):
            return ages, "quote timestamp unknown"
        quote_ts = quote.ts if quote.ts.tzinfo is not None else quote.ts.replace(tzinfo=UTC)
        age = (now - quote_ts.astimezone(UTC)).total_seconds()
        ages[spec] = age
        if age < 0:
            return ages, "quote timestamp is in the future"
        if max_age_seconds > 0 and age > max_age_seconds:
            return ages, f"quotes are stale (oldest {age:.0f}s > {max_age_seconds}s)"
    return ages, None


def _day_start(now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=UTC)


def _mark_exit_request(
    engine: Engine,
    request_id: int,
    status: str,
    decision_reason: str,
    now: datetime,
) -> bool:
    """Conditionally terminalize an unclaimed request without overwriting races."""
    with engine.begin() as conn:
        result = conn.execute(
            update(exit_requests)
            .where(exit_requests.c.id == request_id)
            .where(exit_requests.c.status == "requested")
            .values(
                status=status,
                decision_reason=decision_reason,
                processed_at=now,
            )
        )
    return result.rowcount == 1


def _bind_exit_request(
    engine: Engine,
    request_id: int,
    position_id: int,
    close_order_id: int,
    decision_reason: str,
) -> bool:
    """Persist exact Hermes attribution before any broker mutation."""
    with engine.begin() as conn:
        result = conn.execute(
            update(exit_requests)
            .where(exit_requests.c.id == request_id)
            .where(exit_requests.c.position_id == position_id)
            .where(exit_requests.c.status == "requested")
            .where(exit_requests.c.close_order_id.is_(None))
            .values(
                close_order_id=close_order_id,
                decision_reason=decision_reason,
            )
        )
    return result.rowcount == 1


def _bound_exit_request_is_eligible(
    engine: Engine,
    request_id: int,
    position_id: int,
    close_order_id: int,
    decision_reason: str,
) -> bool:
    """Re-confirm the bound request after the last awaited safety check."""
    with engine.begin() as conn:
        result = conn.execute(
            update(exit_requests)
            .where(exit_requests.c.id == request_id)
            .where(exit_requests.c.position_id == position_id)
            .where(exit_requests.c.status == "requested")
            .where(exit_requests.c.close_order_id == close_order_id)
            .values(decision_reason=decision_reason)
        )
    return result.rowcount == 1


def _finish_bound_exit_request(
    engine: Engine,
    request_id: int,
    close_order_id: int,
    status: str,
    decision_reason: str,
    now: datetime,
) -> bool:
    """Conditionally terminalize the exact request/close binding."""
    with engine.begin() as conn:
        result = conn.execute(
            update(exit_requests)
            .where(exit_requests.c.id == request_id)
            .where(exit_requests.c.status == "requested")
            .where(exit_requests.c.close_order_id == close_order_id)
            .values(
                status=status,
                decision_reason=decision_reason,
                processed_at=now,
            )
        )
    return result.rowcount == 1


def _request_counts(engine: Engine, position_id: int, now: datetime) -> tuple[int, int]:
    start = _day_start(now)
    with engine.connect() as conn:
        position_count = conn.execute(
            select(exit_requests.c.id)
            .where(exit_requests.c.position_id == position_id)
            .where(exit_requests.c.requested_at >= start)
            .where(exit_requests.c.status == "submitted")
        ).fetchall()
        portfolio_count = conn.execute(
            select(exit_requests.c.id)
            .where(exit_requests.c.requested_at >= start)
            .where(exit_requests.c.status == "submitted")
        ).fetchall()
    return len(position_count), len(portfolio_count)


async def _quote_gate_state(
    context: DaemonContext,
    md: MarketDataClient,
    entry: OrderRecord,
    now: datetime,
) -> tuple[QuoteGateState | None, str | None]:
    total_premium = net_premium(context.engine, entry.id)
    if total_premium is None or entry.quantity < 1:
        return None, "entry premium unavailable"
    entry_net = total_premium / (100 * entry.quantity)
    dte = _min_dte(entry.legs, now)
    option_legs = [leg for leg in entry.legs if leg.get("sec_type", "OPT") == "OPT"]
    quotes: dict[tuple[str, float, str], OptionQuote] = {}
    current_net: float | None = None
    try:
        for leg in option_legs:
            spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
            quotes[spec] = await md.get_option_snapshot(
                entry.symbol,
                spec[0],
                spec[1],
                spec[2],  # type: ignore[arg-type]
            )
        current_net = combo_mid(entry.legs, quotes)
    except Exception:  # noqa: BLE001 -- request_exit fails closed without quote corroboration
        log.warning("request_exit quote gate failed for entry #%s", entry.id)
        quotes = {}
    _, quote_issue = _exit_quote_readiness(
        quotes,
        datetime.now(UTC),
        context.settings.execution.exit_quote_max_age_seconds,
    )
    if current_net is None or quote_issue is not None:
        current_net = None
    deterministic_reason = None
    if dte is not None:
        deterministic_reason = evaluate_exit(
            entry_net=entry_net,
            current_net=current_net,
            dte=dte,
            settings=context.settings,
            minutes_to_close=minutes_to_nyse_close(now),
        )
    return QuoteGateState(
        entry_net=entry_net,
        current_net=current_net,
        dte=dte,
        deterministic_exit_reason=deterministic_reason,
    ), None


def _hermes_loss_cap_decision(
    context: DaemonContext,
    now: datetime,
) -> HermesLossCapDecision:
    session = nyse_session_date(now).isoformat()
    session_start = nyse_session_start_utc(now)
    with context.engine.connect() as conn:
        baseline_row = conn.execute(
            select(
                execution_state.c.day_start_net_liq,
                execution_state.c.day_start_session,
            ).where(execution_state.c.id == 1)
        ).first()
        close_rows = conn.execute(
            select(exit_requests.c.close_order_id).where(
                exit_requests.c.close_order_id.is_not(None)
            )
        ).fetchall()
    baseline = None
    if baseline_row is not None and baseline_row.day_start_session == session:
        baseline = baseline_row.day_start_net_liq
    hermes_close_ids = {int(row.close_order_id) for row in close_rows}
    try:
        cumulative_pnl = sum(
            pair.pnl
            for pair in realized_close_pairs(context.engine, since=session_start)
            if pair.close_id in hermes_close_ids
        )
    except RealizedPnLUnavailable as exc:
        return HermesLossCapDecision(
            allowed=False,
            evaluable=False,
            cumulative_realized_pnl=0.0,
            cap_dollars=None,
            reason=f"Hermes realized P&L accounting unavailable: {exc}",
        )
    return evaluate_hermes_loss_cap(
        cumulative_realized_pnl=cumulative_pnl,
        day_start_net_liq=None if baseline is None else float(baseline),
        max_daily_loss_pct=context.settings.execution.max_daily_loss_pct,
    )


async def _enforce_hermes_loss_cap(
    context: DaemonContext,
    now: datetime,
) -> HermesLossCapDecision:
    """Evaluate the cap every tick, even after the final position closes.

    A breached, evaluable cap trips the persisted kill switch once. Any queued
    Hermes requests are refused for both a breach and an unavailable cap; the
    deterministic exit path remains independent when the cap is unevaluable.
    """
    loss_cap = _hermes_loss_cap_decision(context, now)
    if loss_cap.allowed:
        return loss_cap

    if loss_cap.evaluable and not load_state(context.engine).killed:
        trip_kill(context.engine, loss_cap.reason, now=now)
        await _send(
            context,
            "🛑 HALT: cumulative Hermes-driven realized losses breached the daily cap. "
            f"{loss_cap.reason}. No further Hermes exit requests will be processed.",
        )

    with context.engine.connect() as conn:
        pending_ids = [
            int(row.id)
            for row in conn.execute(
                select(exit_requests.c.id)
                .where(exit_requests.c.status == "requested")
                .order_by(exit_requests.c.requested_at)
            ).fetchall()
        ]
    for request_id in pending_ids:
        _mark_exit_request(
            context.engine,
            request_id,
            "refused",
            loss_cap.reason,
            now,
        )
    return loss_cap


async def _process_exit_requests(
    context: DaemonContext,
    md: MarketDataClient,
    entries: list[OrderRecord],
    now: datetime,
) -> int:
    by_id = {entry.id: entry for entry in entries}
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(exit_requests)
            .where(exit_requests.c.status == "requested")
            .order_by(exit_requests.c.requested_at)
        ).fetchall()
    submitted = 0
    for row in rows:
        entry = by_id.get(row.position_id)
        if entry is None:
            _mark_exit_request(context.engine, row.id, "refused", "position is no longer open", now)
            continue
        if open_close_for(context.engine, entry.id) is not None:
            _mark_exit_request(
                context.engine, row.id, "refused", "position is already closing", now
            )
            continue
        raw_sources = row.sources_json
        try:
            confidence = float(row.confidence)
        except (TypeError, ValueError, OverflowError):
            confidence = math.nan
        sources = (
            [source.strip() for source in raw_sources]
            if isinstance(raw_sources, list)
            and all(isinstance(source, str) for source in raw_sources)
            else []
        )
        canonical_sources = {source.casefold() for source in sources}
        reason = row.reason.strip() if isinstance(row.reason, str) else ""
        catalyst_type = row.catalyst_type.strip() if isinstance(row.catalyst_type, str) else ""
        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or len(sources) < 2
            or any(not source for source in sources)
            or len(canonical_sources) != len(sources)
            or not reason
            or not catalyst_type
        ):
            _mark_exit_request(
                context.engine,
                row.id,
                "refused",
                "persisted exit authorization evidence is invalid",
                now,
            )
            continue
        state, unavailable = await _quote_gate_state(context, md, entry, now)
        if state is None:
            _mark_exit_request(context.engine, row.id, "refused", unavailable or "unavailable", now)
            continue
        position_count, portfolio_count = _request_counts(context.engine, entry.id, now)
        decision = evaluate_exit_request_gate(
            ExitRequestGateInput(
                position_id=entry.id,
                catalyst_type=catalyst_type,
                confidence=confidence,
                sources=sources,
                reason=reason,
                today_position_requests=position_count,
                today_portfolio_requests=portfolio_count,
            ),
            state,
        )
        if not decision.allowed:
            _mark_exit_request(context.engine, row.id, "refused", decision.reason, now)
            continue
        placed = await _manage_entry(
            context,
            md,
            entry,
            now,
            forced_reason=f"Hermes request_exit #{row.id}: {decision.reason}",
            exit_request_id=int(row.id),
            exit_decision_reason=decision.reason,
        )
        if placed:
            submitted += placed
        else:
            _mark_exit_request(context.engine, row.id, "failed", "no close was placed", now)
    return submitted


async def run_exits_tick(context: DaemonContext) -> ExitsTickSummary:
    now = datetime.now(UTC)
    await _enforce_hermes_loss_cap(context, now)
    if context.order_client is None:
        return ExitsTickSummary(0, 0, 0)
    entries = _open_entries(context)
    if not entries:
        return ExitsTickSummary(0, 0, 0)

    submitted = errors = 0
    entry_verdict = can_execute(context.settings, load_state(context.engine))
    exit_verdict = can_reduce_risk(context.settings)
    md = _exec_md(context)
    # Protective TP / soft-stop / DTE / expiry closes deliberately ignore the
    # kill bit: a drawdown halt blocks NEW risk, never risk reduction. Broker
    # environment/config and market-hours interlocks remain mandatory. Hermes
    # discretionary exit requests retain the stricter entry gate while killed.
    if exit_verdict.allowed and md is not None and is_market_open(now):
        if entry_verdict.allowed:
            try:
                submitted += await _process_exit_requests(context, md, entries, now)
            except Exception:  # noqa: BLE001 -- request queue must not starve deterministic exits
                errors += 1
                log.exception("request_exit queue processing failed")
        for entry in entries:
            try:
                submitted += await _manage_entry(context, md, entry, now)
            except Exception:  # noqa: BLE001 -- one bad position must not starve the rest
                errors += 1
                log.exception("exit evaluation failed for entry #%s", entry.id)
    elif not exit_verdict.allowed:
        log.warning("exits tick: protective placement disabled: %s", exit_verdict.reason)
    # Post-close naked-short P1 sweep: detect-and-halt (trip the kill + alert, no
    # order placement). It reads only the broker portfolio, so it runs on EVERY
    # tick regardless of the interlock, market hours, or quote availability
    # (IBK-142 freed it from the market-hours gate; IBK-145 from can_execute + md)
    # -- a residual naked short is most dangerous exactly when the account is
    # halted or the market is closed.
    for entry in entries:
        try:
            if open_close_for(context.engine, entry.id) is None and _half_closed(
                context.engine, entry.id
            ):
                # A close for this entry has terminated with partial fills:
                # confirm the broker isn't sitting on a naked short.
                await assert_no_naked_short_after_close(context, entry)
        except Exception:  # noqa: BLE001 -- safety sweep is best-effort
            log.exception("post-close naked-leg sweep failed for #%s", entry.id)
    if entries:
        log.info(
            "exits tick: positions=%d closes=%d errors=%d",
            len(entries),
            submitted,
            errors,
        )
    return ExitsTickSummary(positions=len(entries), closes_submitted=submitted, errors=errors)


async def _manage_entry(
    context: DaemonContext,
    md: MarketDataClient,
    entry: OrderRecord,
    now: datetime,
    *,
    forced_reason: str | None = None,
    exit_request_id: int | None = None,
    exit_decision_reason: str | None = None,
) -> int:
    """Evaluate one position; returns 1 if a closing order was submitted.

    ``forced_reason`` bypasses ``evaluate_exit`` for an approved manual or
    Hermes-triggered close. Hermes requests additionally carry their exact
    persisted identity so attribution is bound before broker placement.
    """
    engine = context.engine
    order_client = context.order_client
    if order_client is None:
        return 0
    execution_verdict = (
        can_execute(context.settings, load_state(engine))
        if exit_request_id is not None
        else can_reduce_risk(context.settings)
    )
    if not execution_verdict.allowed:
        if exit_request_id is not None:
            _mark_exit_request(
                engine,
                exit_request_id,
                "refused",
                f"execution interlock closed: {execution_verdict.reason}",
                now,
            )
        return 0
    if open_close_for(engine, entry.id) is not None:
        return 0  # a close is already working — never double-exit
    if _half_closed(engine, entry.id):
        # A close partially filled and then died (abandoned/cancelled): the
        # broker position is HALF flat. Re-staging a full-quantity close
        # would over-close into wrong-way exposure — hand off to the human.
        if entry.id not in context.exit_handoff_warned:
            context.exit_handoff_warned.add(entry.id)
            await _send(
                context,
                f"⚠ position #{entry.id} {entry.symbol} {entry.strategy} is "
                "HALF-CLOSED (a closing order partially filled, then died). "
                "Auto-exit is halted for this position — flatten the remainder "
                "manually in TWS.",
            )
        return 0
    dte = _min_dte(entry.legs, now)
    if dte is None and forced_reason is None:
        return 0

    total_premium = net_premium(engine, entry.id)
    if total_premium is None or entry.quantity < 1:
        return 0
    entry_net = total_premium / (100 * entry.quantity)  # per-unit, signed

    option_legs = [leg for leg in entry.legs if leg.get("sec_type", "OPT") == "OPT"]
    quotes: dict[tuple[str, float, str], OptionQuote] = {}
    current_net: float | None = None
    try:
        for leg in option_legs:
            spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
            quotes[spec] = await md.get_option_snapshot(
                entry.symbol,
                spec[0],
                spec[1],
                spec[2],  # type: ignore[arg-type]
            )
        current_net = combo_mid(entry.legs, quotes)
    except Exception:  # noqa: BLE001 -- deterministic exits must still work quote-blind
        log.warning("exit quotes failed for entry #%s — DTE rules only", entry.id)
        quotes = {}

    max_age = context.settings.execution.exit_quote_max_age_seconds
    quote_ages, quote_issue = _exit_quote_readiness(quotes, datetime.now(UTC), max_age)
    if quote_issue is None and current_net is None:
        quote_issue = "quote mid unavailable"
    if quotes and quote_issue is None:
        log.info(
            "exit quotes for entry #%s: %s",
            entry.id,
            ", ".join(
                f"{spec[1]}{spec[2]}@{q.ts.isoformat()} (age {quote_ages[spec]:.0f}s)"
                for spec, q in quotes.items()
                if q.ts is not None
            ),
        )
    if quote_issue is not None:
        # Keep the existing one-shot operator alert for stale delivery. Other
        # unavailable shapes are logged and fail closed without quote-driven
        # TP/stop evaluation; deterministic DTE/expiry rules remain active.
        if (
            "stale" in quote_issue
            and forced_reason is None
            and entry.id not in context.exit_stale_warned
        ):
            context.exit_stale_warned.add(entry.id)
            oldest = max(quote_ages.values(), default=0.0)
            await _send(
                context,
                f"⚠ exit pricing for #{entry.id} {entry.symbol} {entry.strategy} "
                f"suppressed: quotes are STALE (oldest {oldest:.0f}s > "
                f"{max_age}s). TP/stop deferred; expiry/DTE rules still active.",
            )
        log.warning(
            "exit #%s quote-priced exit suppressed (%s)",
            entry.id,
            quote_issue,
        )
        current_net = None
        quotes = {}  # unusable quotes must not price the close or a walk
    else:
        context.exit_stale_warned.discard(entry.id)

    reason: str | None
    if forced_reason is not None:
        reason = forced_reason
    else:
        # The dte early-return above only let a None dte through on the forced
        # path, so dte is non-None here (narrow for evaluate_exit).
        assert dte is not None
        reason = evaluate_exit(
            entry_net=entry_net,
            current_net=current_net,
            dte=dte,
            settings=context.settings,
            minutes_to_close=minutes_to_nyse_close(now),
        )
    if reason is None:
        return 0

    # Price the FLIPPED structure. Closing a credit pays; closing a debit
    # collects. A deterministic time/DTE close with no usable live quote falls
    # back to the entry net and does not start a quote-driven price walk.
    # Rounded to the symbol tick: IBKR rejects sub-increment limits (Error 110).
    increment = price_increment_for(entry.symbol)
    close_net = round_to_increment(
        -(current_net if current_net is not None else entry_net), increment
    )
    limit_price = -close_net  # BAG convention: limit = -net

    try:
        close = stage_close_order(engine, entry, now=now)
    except CloseAlreadyClaimed as exc:
        if exit_request_id is not None:
            _mark_exit_request(engine, exit_request_id, "refused", str(exc), now)
        return 0
    try:
        assert_atomic_close_legs(entry_legs=entry.legs, close_legs=close.legs)
    except NonAtomicCloseError as exc:
        # Fail safe: a close we cannot guarantee atomic must NOT be legged out.
        transition(engine, close.id, "skipped", error=str(exc), now=now)
        if exit_request_id is not None:
            _mark_exit_request(engine, exit_request_id, "failed", str(exc), now)
        trip_kill(engine, f"non-atomic close for #{entry.id}: {exc}")
        await _send(
            context,
            f"🛑 HALT: close for #{entry.id} {entry.symbol} {entry.strategy} is "
            f"NOT an atomic combo ({exc}). Kill switch tripped — no order placed. "
            "Flatten manually and /arm after fixing.",
        )
        return 0

    if exit_request_id is not None:
        if exit_decision_reason is None:
            transition(engine, close.id, "skipped", error="missing Hermes decision", now=now)
            _mark_exit_request(
                engine,
                exit_request_id,
                "failed",
                "missing Hermes decision attribution",
                now,
            )
            return 0
        if not _bind_exit_request(
            engine,
            exit_request_id,
            entry.id,
            close.id,
            exit_decision_reason,
        ):
            transition(engine, close.id, "skipped", error="exit request claim lost", now=now)
            return 0

        # This is the last awaited gate before placement. Recompute from the
        # complete bound-close ledger for every Hermes broker mutation, rather
        # than relying on the once-per-tick value.
        loss_cap = await _enforce_hermes_loss_cap(context, now)
        if not loss_cap.allowed:
            transition(engine, close.id, "skipped", error=loss_cap.reason, now=now)
            _finish_bound_exit_request(
                engine,
                exit_request_id,
                close.id,
                "refused",
                loss_cap.reason,
                now,
            )
            return 0
        if not _bound_exit_request_is_eligible(
            engine,
            exit_request_id,
            entry.id,
            close.id,
            exit_decision_reason,
        ):
            transition(
                engine,
                close.id,
                "skipped",
                error="exit request no longer eligible",
                now=now,
            )
            return 0

    execution_verdict = (
        can_execute(context.settings, load_state(engine))
        if exit_request_id is not None
        else can_reduce_risk(context.settings)
    )
    if not execution_verdict.allowed:
        reason = f"execution interlock closed before placement: {execution_verdict.reason}"
        transition(engine, close.id, "skipped", error=reason, now=now)
        if exit_request_id is not None:
            _finish_bound_exit_request(
                engine,
                exit_request_id,
                close.id,
                "refused",
                reason,
                now,
            )
        return 0

    transition(engine, close.id, "submitting", now=now)
    try:
        placed = await order_client.place_combo_limit(
            entry.symbol,
            close.legs,
            quantity=close.quantity,
            limit_price=limit_price,
            order_ref=close.order_ref or f"obot-{close.id}",
        )
    except Exception as exc:  # noqa: BLE001 -- placement outcome may be unknown
        halt_reason = f"close #{close.id} broker placement outcome unknown after exception: {exc}"
        # Keep the close in `submitting` so the permanent active-close claim
        # blocks retries until broker/ledger reconciliation determines whether
        # the order exists. A terminal transition here could stage a duplicate.
        try:
            set_order_note(engine, close.id, halt_reason)
        except Exception:  # noqa: BLE001 -- kill still takes precedence
            log.exception("failed to annotate uncertain close #%s", close.id)
        trip_kill(engine, halt_reason, now=now)
        await _send(
            context,
            f"🛑 HALT: {halt_reason}. No retry will be staged; reconcile broker "
            "and ledger state before re-arming.",
        )
        return 0
    try:
        set_order_leg_contracts(engine, close.id, placed.leg_contracts)
        transition(
            engine,
            close.id,
            "submitted",
            ib_order_id=placed.ib_order_id,
            now=now,
        )
    except Exception as exc:  # noqa: BLE001 -- broker side effect may already exist
        halt_reason = (
            f"close #{close.id} broker placement completed but ledger finalization failed: {exc}"
        )
        trip_kill(engine, halt_reason, now=now)
        try:
            await order_client.cancel(placed.ib_order_id)
        except Exception:  # noqa: BLE001 -- halt remains authoritative
            log.exception(
                "failed to cancel close #%s after ledger finalization failure",
                close.id,
            )
        await _send(
            context,
            f"🛑 HALT: {halt_reason}. Treat broker state as unknown and reconcile "
            "before re-arming.",
        )
        return 1

    if exit_request_id is not None and exit_decision_reason is not None:
        try:
            completed = _finish_bound_exit_request(
                engine,
                exit_request_id,
                close.id,
                "submitted",
                exit_decision_reason,
                now,
            )
        except Exception as exc:  # noqa: BLE001 -- broker side effect already exists
            halt_reason = (
                f"Hermes exit request #{exit_request_id} completion failed after "
                f"broker placement of close #{close.id}: {exc}"
            )
            trip_kill(engine, halt_reason, now=now)
            await _send(
                context,
                f"🛑 HALT: {halt_reason}. The close remains broker-submitted; "
                "reconcile before re-arming.",
            )
            return 1
        if not completed:
            halt_reason = (
                f"Hermes exit request #{exit_request_id} completion lost after "
                f"broker placement of close #{close.id}"
            )
            trip_kill(engine, halt_reason, now=now)
            await _send(
                context,
                f"🛑 HALT: {halt_reason}. The close remains broker-submitted; "
                "no further automated mutations will be started.",
            )
            return 1

    # Walk only when this close was initially priced from a complete live,
    # current quote set. Quote-blind deterministic closes must never re-anchor
    # to delayed/unknown data in the generic walk implementation.
    cfg = context.settings.execution
    if cfg.walk_max_steps > 0 and current_net is not None:
        flipped_nbbo = combo_bid_ask(close.legs, quotes) if quotes else None
        budget = (
            slippage_budget(
                flipped_nbbo[0],
                flipped_nbbo[1],
                frac=cfg.max_slippage_spread_frac,
                abs_cap=cfg.max_slippage_abs,
                increment=increment,
            )
            if flipped_nbbo is not None
            else increment
        )
        from optionsbot.execution.walk import run_price_walk

        task = asyncio.create_task(
            run_price_walk(
                engine=engine,
                settings=context.settings,
                order_client=order_client,
                md=md,
                symbol=entry.symbol,
                legs=close.legs,
                order_id=close.id,
                ib_order_id=placed.ib_order_id,
                decision_mid=close_net,
                budget=budget,
                increment=increment,
            )
        )
        context.walk_tasks.add(task)
        task.add_done_callback(context.walk_tasks.discard)

    await _send(
        context,
        f"📤 closing #{entry.id} {entry.symbol} {entry.strategy} "
        f"{entry.quantity}x — {reason} (close order #{close.id})",
    )
    if reason.startswith("soft stop") and context.events is not None:
        context.events.emit(
            "stop-hit",
            f"Soft stop submitted for #{entry.id} {entry.symbol} {entry.strategy}",
            severity="warning",
            details={"entry_order_id": entry.id, "close_order_id": close.id, "reason": reason},
        )
    return 1


async def force_close_entry(
    context: DaemonContext, entry_id: int, now: datetime | None = None
) -> str:
    """Human-initiated close of one ledger-attributed open position (``/close``).

    Runs the SAME close path as the exit engine (``_manage_entry``) but with the
    human as the trigger instead of ``evaluate_exit`` — every guard (atomic
    combo, half-closed, double-close) and the price-walk are reused untouched.
    Returns a status line for the Telegram reply; the rich "closing #N" line is
    sent by ``_manage_entry`` itself.
    """
    now = now if now is not None else datetime.now(UTC)
    engine = context.engine
    if context.order_client is None:
        return "execution is not configured in this daemon build"
    entry = get_order(engine, entry_id)
    if entry is None:
        return f"unknown order id {entry_id}"
    if entry.intent != "open" or entry.status != "filled":
        return (
            f"#{entry_id} is {entry.intent}/{entry.status} — /close only acts on a "
            "filled open position (see /orders)"
        )
    # Double-close guard. Safe without a lock only because neither this function
    # nor _manage_entry awaits between this check and stage_close_order/transition
    # — the cadence exits tick sees the resulting non-terminal close row. Keep it
    # that way: any await inserted before staging reopens a double-close race.
    if open_close_for(engine, entry_id) is not None:
        return f"#{entry_id} {entry.symbol} {entry.strategy} is already closing"
    verdict = can_reduce_risk(context.settings)
    if not verdict.allowed:
        return f"can't close #{entry_id}: {verdict.reason}"
    if not is_market_open(now):
        return f"market is closed — /close #{entry_id} would not fill; try during RTH"
    md = _exec_md(context)
    if md is None:
        return f"can't close #{entry_id}: exec market-data connection unavailable"
    submitted = await _manage_entry(
        context, md, entry, now, forced_reason="manual /close via Telegram"
    )
    if submitted:
        return (
            f"close requested for #{entry_id} {entry.symbol} {entry.strategy} "
            f"{entry.quantity}x — walking to fill; you'll get the confirmation"
        )
    return (
        f"#{entry_id} {entry.symbol} {entry.strategy}: no close placed — it may be "
        "half-closed or unpriceable (check /orders)"
    )


async def assert_no_naked_short_after_close(context: DaemonContext, entry: OrderRecord) -> bool:
    """After a close fills, verify no SHORT leg of this entry is still open at
    the broker. A residual naked short is a P1 incident: trip the kill switch
    and alert (once). Returns True when clean, False when a naked short was
    found. Never raises (a broker read failure is logged, not fatal)."""
    from optionsbot.ibkr.positions import PositionsClient

    if context.exec_ibkr is None:
        reason = f"post-close broker position verification unavailable for #{entry.id}"
        trip_kill(context.engine, reason)
        if entry.id not in context.naked_leg_halted:
            context.naked_leg_halted.add(entry.id)
            await _send(context, f"🛑 KILL SWITCH: {reason}")
        return False
    try:
        positions = await PositionsClient(context.exec_ibkr).get_portfolio()
        if not isinstance(positions, (list, tuple)):
            raise ValueError("portfolio snapshot is not a list or tuple")
        naked = find_naked_short_legs(entry.legs, positions)
    except Exception as exc:  # noqa: BLE001 -- unavailable/malformed state fails closed
        log.exception("naked-leg check: portfolio evidence failed for #%s", entry.id)
        reason = f"post-close broker position verification failed for #{entry.id}: {exc}"
        trip_kill(context.engine, reason)
        if entry.id not in context.naked_leg_halted:
            context.naked_leg_halted.add(entry.id)
            await _send(context, f"🛑 KILL SWITCH: {reason}")
        return False
    if not naked:
        context.naked_leg_halted.discard(entry.id)
        return True
    if entry.id not in context.naked_leg_halted:
        context.naked_leg_halted.add(entry.id)
        trip_kill(context.engine, f"naked short leg after close for #{entry.id}")
        legs_desc = ", ".join(f"{p.strike}{p.right} x{p.position:.0f}" for p in naked)
        await _send(
            context,
            f"🛑 P1: #{entry.id} {entry.symbol} {entry.strategy} has a NAKED "
            f"SHORT leg after its close ({legs_desc}). Kill switch tripped — "
            "re-hedge or flatten this leg in TWS immediately, then /arm.",
        )
    log.error("naked short after close for entry #%s: %s", entry.id, naked)
    return False


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001
        log.exception("exit notification failed")
