"""/execute orchestration (IBK-126): gates → fresh pricing → stage → place.

Every rejection returns a human-readable reason (the Telegram reply IS the
UX in confirm mode). Pricing is v1: place at the fresh combo mid and let the
order watcher TTL-cancel if unfilled — the reprice ladder is IBK-127.

NOT re-exported from ``optionsbot.execution.__init__`` (imports ibkr +
daemon submodules; same import-graph reasoning as tracker.py). Callers
import it directly inside functions — see daemon/commands.py.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError

from optionsbot.config import Settings
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.execution.gate import can_execute
from optionsbot.execution.orders import (
    record_order_quotes,
    set_order_leg_contracts,
    set_order_note,
    stage_order,
    transition,
)
from optionsbot.execution.risk_structure import (
    has_structurally_defined_option_risk,
    structural_max_loss_dollars,
)
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.execution.walk import (
    combo_bid_ask,
    combo_spread_issue,
    liquidity_issues,
    price_increment_for,
    round_to_increment,
    run_price_walk,
    slippage_budget,
)
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.orders import OrderClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.ibkr.types import OptionQuote
from optionsbot.review_evidence import snapshot_ready_for_auto
from optionsbot.storage.schema import orders, snapshots, strategy_scores

log = logging.getLogger(__name__)

# Statuses that count as "this pick already has live exposure": anything not
# failed-terminal. `filled` stays active until IBK-129 introduces closes.
_ACTIVE_STATUSES = ("staged", "submitting", "submitted", "partial", "filled")

LegSpec = tuple[str, float, str]


@dataclass(frozen=True, slots=True)
class ExecutionDeps:
    """Everything execute_pick needs; daemon/commands assembles it from context.

    walk_md is a MarketDataClient bound to the EXEC connection (the walk
    re-anchors quotes without ibkr_lock); walk_tasks holds strong refs to
    spawned walk tasks. Either being None disables the walk (v1 behavior:
    rest at mid until the TTL watcher cancels).
    """

    engine: Engine
    settings: Settings
    order_client: OrderClient
    md: MarketDataClient
    positions: PositionsClient
    ibkr_lock: asyncio.Lock
    walk_md: MarketDataClient | None = None
    walk_tasks: set[asyncio.Task[None]] | None = None


@dataclass(frozen=True, slots=True)
class ExecuteOutcome:
    ok: bool
    message: str
    order_id: int | None = None


def combo_mid(
    legs: Sequence[Mapping[str, Any]], quotes: Mapping[LegSpec, OptionQuote]
) -> float | None:
    """Signed per-unit net mid from per-leg quotes: positive = net credit.

    STK legs are ignored (never ordered). Returns None unless EVERY option
    leg has a usable mid — a partial mid is a wrong price, not a fallback.
    """
    total = 0.0
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
        quote = quotes.get(spec)
        if quote is None or quote.mid is None:
            return None
        sign = 1.0 if leg["side"] == "sell" else -1.0
        total += sign * quote.mid * int(leg.get("quantity", 1))
    return total


def _reject(message: str) -> ExecuteOutcome:
    # Log every rejection reason to the journal. Without this, execute_pick's
    # gate decisions were only visible in the Telegram outcome message (the
    # daemon's auto-execute path sends them to chat, not the log), so a pick
    # that silently failed a gate left no trace in `journalctl` -- which made
    # "why didn't it trade?" undiagnosable from the logs.
    log.info("execute_pick reject: %s", message)
    return ExecuteOutcome(ok=False, message=f"❌ {message}")


async def execute_pick(
    deps: ExecutionDeps, score_id: int, *, now: datetime | None = None
) -> ExecuteOutcome:
    ts_now = now if now is not None else datetime.now(UTC)
    engine = deps.engine
    settings = deps.settings

    # 1. Arming gate (enabled + paper interlock + kill switch).
    verdict = can_execute(settings, load_state(engine))
    if not verdict.allowed:
        return _reject(verdict.reason)

    # PHASE 0 B1: stop ADDING risk as the day-start net-liq drawdown approaches
    # the daily-loss cap. The per-tick breaker is the hard kill; this is the
    # softer "don't pile on while bleeding" entry gate. Missing or invalid
    # net-liq fails closed because the breaker cannot prove safety.
    from optionsbot.execution.equity_guard import new_entry_allowed

    try:
        async with deps.ibkr_lock:
            _summary = await deps.positions.get_account_summary()
        _net_liq = (
            float(_summary.net_liquidation_usd)
            if _summary.net_liquidation_usd is not None
            else None
        )
    except Exception:  # noqa: BLE001 -- unreadable equity fails closed below
        _net_liq = None
    _entry = new_entry_allowed(engine, settings, current_net_liq=_net_liq)
    if not _entry.allowed:
        return _reject(_entry.reason)

    # 2. Load the pick.
    with engine.connect() as conn:
        pick = conn.execute(
            select(
                strategy_scores.c.id,
                strategy_scores.c.strategy,
                strategy_scores.c.legs_json,
                strategy_scores.c.suggestion_json,
                snapshots.c.symbol,
                snapshots.c.ts,
            )
            .add_columns(snapshots.c.raw_json)
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(strategy_scores.c.id == score_id)
        ).first()
    if pick is None:
        return _reject(f"unknown pick id {score_id}")
    suggestion: dict[str, Any] = pick.suggestion_json or {}
    legs: list[dict[str, Any]] = list(pick.legs_json or [])
    symbol: str = pick.symbol

    # 3. Freshness — stale strikes are the wrong trade.
    pick_ts: datetime = pick.ts
    if pick_ts.tzinfo is None:
        pick_ts = pick_ts.replace(tzinfo=UTC)
    age = ts_now - pick_ts
    max_age = timedelta(minutes=settings.execution.max_pick_age_minutes)
    if age > max_age:
        return _reject(
            f"stale pick — scanned {int(age.total_seconds() // 60)}m ago "
            f"(max {settings.execution.max_pick_age_minutes}m). /scan {symbol} for a fresh one."
        )
    if age < timedelta(0):
        return _reject("future-dated pick — persisted scan time is not trustworthy")

    # 4. Defined-risk only. In auto mode, earnings inside the expiry window
    # are skipped — binary jump risk overwhelms theta edge for neutral
    # structures (the human can still /execute deliberately in confirm mode).
    if settings.execution.mode == "auto":
        snapshot_raw = pick.raw_json
        if not isinstance(snapshot_raw, dict):
            return _reject(
                "snapshot data readiness unavailable — auto mode requires audited live data"
            )
        if not snapshot_ready_for_auto(snapshot_raw):
            return _reject(
                "snapshot data not explicitly live and ready — auto mode requires "
                "delayed=false; warming up requires an explicit HV-rank proxy"
            )
        if (
            settings.execution.auto_skip_earnings
            and snapshot_raw.get("earnings_in_window")
        ):
            return _reject(
                "earnings inside the expiry window — auto mode skips "
                "(use /execute to override deliberately)"
            )
    if (
        suggestion.get("defined_risk") is not True
        or not has_structurally_defined_option_risk(legs)
    ):
        return _reject("undefined risk: option legs do not independently prove a bound")
    max_loss_unit = float(suggestion.get("max_loss") or 0.0)
    if not math.isfinite(max_loss_unit) or max_loss_unit <= 0:
        return _reject("pick carries no defined max loss — not sizeable")

    # 5. Market hours.
    if not is_market_open(ts_now):
        return _reject("market is closed — orders would rest blind on stale quotes")

    # 6. One active order per pick.
    with engine.connect() as conn:
        existing = conn.execute(
            select(orders.c.id, orders.c.status)
            .where(orders.c.strategy_score_id == score_id)
            .where(orders.c.status.in_(_ACTIVE_STATUSES))
        ).first()
        if existing is not None:
            return _reject(
                f"pick {score_id} already has order #{existing.id} ({existing.status})"
            )
        # 7. Ledger-based caps (bot-attributed exposure; portfolio-wide gates
        # arrive with full-auto IBK-130). Entries whose close has FILLED are
        # round-trips, not exposure — without this exclusion a cycled symbol
        # would be permanently barred and the global cap would ratchet down
        # over the account's lifetime (Opus IBK-130 #1).
        raw_active = conn.execute(
            select(orders.c.id, orders.c.symbol)
            .where(orders.c.intent == "open")
            .where(orders.c.status.in_(_ACTIVE_STATUSES))
        ).fetchall()
        closed_entry_ids = {
            row.closes_order_id
            for row in conn.execute(
                select(orders.c.closes_order_id)
                .where(orders.c.intent == "close")
                .where(orders.c.status == "filled")
            ).fetchall()
        }
    active_rows = [row for row in raw_active if row.id not in closed_entry_ids]
    if len(active_rows) >= settings.execution.max_open_positions:
        return _reject(
            f"max_open_positions reached ({settings.execution.max_open_positions}) "
            "— close something first"
        )
    same_symbol = sum(1 for r in active_rows if r.symbol == symbol)
    if same_symbol >= settings.execution.max_per_symbol:
        return _reject(
            f"already {same_symbol} active {symbol} position(s) "
            f"(max_per_symbol={settings.execution.max_per_symbol})"
        )

    # 8. Fresh per-leg quotes -> combo mid (the scan-time credit is stale).
    option_legs = [leg for leg in legs if leg.get("sec_type", "OPT") == "OPT"]
    quotes: dict[LegSpec, OptionQuote] = {}
    async with deps.ibkr_lock:
        for leg in option_legs:
            spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
            try:
                quotes[spec] = await deps.md.get_option_snapshot(
                    symbol, spec[0], spec[1], spec[2]  # type: ignore[arg-type]
                )
            except Exception as exc:  # noqa: BLE001 -- per-leg quote failure = reject
                return _reject(f"no usable quote for {symbol} {spec[0]} {spec[1]}{spec[2]}: {exc}")
    if any(quote.delayed is not False for quote in quotes.values()):
        return _reject(
            "delayed or unknown option quote feed — refusing to price an entry without live data"
        )
    quote_max_age = timedelta(seconds=settings.execution.entry_quote_max_age_seconds)
    quote_now = ts_now if now is not None else datetime.now(UTC)
    for quote in quotes.values():
        if not isinstance(quote.ts, datetime):
            return _reject("option quote timestamp unknown — refusing to price an entry blind")
        quote_ts = quote.ts if quote.ts.tzinfo is not None else quote.ts.replace(tzinfo=UTC)
        quote_age = quote_now - quote_ts.astimezone(UTC)
        if quote_age < timedelta(0) or quote_age > quote_max_age:
            return _reject(
                "option quote age outside execution window "
                f"({quote_age.total_seconds():.0f}s; max "
                f"{settings.execution.entry_quote_max_age_seconds}s)"
            )
    issues = liquidity_issues(
        legs, quotes,
        leg_spread_frac=settings.execution.max_leg_spread_frac,
        leg_spread_floor=settings.execution.max_leg_spread_floor,
        min_open_interest=settings.execution.min_open_interest,
    )
    if issues:
        message = "liquidity: " + "; ".join(issues)
        if all("no bid/ask" in issue for issue in issues):
            # Every leg empty = almost always IBKR's one-session rule, not
            # illiquidity (Error 10197 — a LIVE login is consuming the feed).
            message += (
                "\n(all quotes empty — usually a competing LIVE login: close "
                "live TWS/Client Portal/mobile and retry)"
            )
        return _reject(message)
    fresh_net = combo_mid(legs, quotes)
    if fresh_net is None:
        return _reject("missing quote mid on at least one leg — not pricing this blind")
    # The raw mid is often a half-cent (bid 9.30/ask 9.33 → 9.315); IBKR
    # rejects sub-increment limits with Error 110. Round to the symbol's tick.
    fresh_net = round_to_increment(fresh_net, price_increment_for(symbol))
    structural_max_loss = structural_max_loss_dollars(
        legs,
        entry_net_per_share=fresh_net,
    )
    if structural_max_loss is None:
        return _reject("option legs do not prove a finite structural max loss")
    # Persisted scanner risk is advisory. Never size below the larger of the
    # scanner estimate and the independently reconstructed expiry loss.
    max_loss_unit = max(max_loss_unit, structural_max_loss)
    # Economic liquidity gate: combo spread vs the premium we're capturing.
    combo_issue = combo_spread_issue(
        legs, quotes, fresh_net, max_frac=settings.execution.max_combo_spread_frac
    )
    if combo_issue is not None:
        return _reject("liquidity: " + combo_issue)

    scan_net = float(suggestion.get("credit_or_debit") or 0.0) / 100.0  # per unit
    if scan_net != 0 and (fresh_net == 0 or (fresh_net > 0) != (scan_net > 0)):
        return _reject(
            f"edge gone — scan priced {scan_net:+.2f}/unit but fresh mid is "
            f"{fresh_net:+.2f}/unit"
        )
    drift_note = ""
    if scan_net != 0:
        drift = abs(fresh_net - scan_net) / abs(scan_net)
        if drift > settings.execution.credit_drift_warn_pct:
            drift_note = (
                f"\n⚠ drift {drift * 100:.0f}% vs scan "
                f"({scan_net:+.2f} → {fresh_net:+.2f})"
            )

    limit_price = -fresh_net  # BUY-bag convention: credit = negative limit

    # 9. Dynamic sizing (IBK-133) from live equity + the bot's own history,
    # then the margin gate via whatIf. Under ibkr_lock: whatif's leg
    # qualification can touch the DAEMON connection (resolver), and all
    # daemon-connection I/O is serialized by discipline.
    async with deps.ibkr_lock:
        summary = await deps.positions.get_account_summary()
    equity = (
        float(summary.net_liquidation_usd)
        if summary.net_liquidation_usd is not None
        else None
    )
    if equity is None or equity <= 0:
        # IBK-122 fail-closed: never size off the stale scan-time quantity (it
        # skips the single-trade/heat caps). A missing live net-liq is rare and
        # transient — reject and let the human re-issue /execute.
        return _reject(
            "live equity unavailable — refusing to size off a stale scan-time quantity"
        )
    final_entry_gate = new_entry_allowed(
        engine, settings, current_net_liq=equity
    )
    if not final_entry_gate.allowed:
        return _reject(final_entry_gate.reason)
    from optionsbot.execution.orders import RealizedPnLUnavailable, realized_close_pairs
    from optionsbot.execution.sizing import dynamic_quantity, open_heat_dollars

    try:
        recent_pnls = [p.pnl for p in realized_close_pairs(engine)[-5:]]
    except RealizedPnLUnavailable as exc:
        return _reject(f"realized P&L accounting unavailable — refusing new risk: {exc}")

    decision = dynamic_quantity(
        equity=equity,
        max_loss_unit=max_loss_unit,
        max_profit_unit=(
            float(suggestion["max_profit"]) if suggestion.get("max_profit") else None
        ),
        prob_profit=(
            float(suggestion["prob_profit"]) if suggestion.get("prob_profit") else None
        ),
        open_heat=open_heat_dollars(engine),
        recent_pnls=recent_pnls,
        base_risk_pct=settings.execution.base_risk_pct,
        heat_cap_pct=settings.execution.max_portfolio_heat_pct,
        single_trade_cap_pct=settings.execution.max_single_trade_risk_pct,
    )
    if decision.quantity < 1:
        return _reject(decision.note)
    quantity = decision.quantity
    sizing_note = decision.note

    async with deps.ibkr_lock:
        try:
            preview = await deps.order_client.whatif_combo(
                symbol, legs, quantity=quantity, limit_price=limit_price
            )
        except Exception as exc:  # noqa: BLE001 -- broker errors become a clean reject
            return _reject(f"whatIf margin preview failed: {exc}")
    execution_verdict = can_execute(settings, load_state(engine))
    if not execution_verdict.allowed:
        return _reject(
            f"execution interlock closed after margin preview: {execution_verdict.reason}"
        )
    available = float(summary.available_funds) if summary.available_funds is not None else None
    # IBK-130: portfolio-wide buying-power deployment cap (auto mode only).
    if settings.execution.mode == "auto":
        net_liq = (
            float(summary.net_liquidation)
            if summary.net_liquidation is not None
            else None
        )
        if net_liq and available is not None and net_liq > 0:
            deployed = (net_liq - available) / net_liq
            if deployed >= settings.execution.max_bp_usage_pct:
                return _reject(
                    f"buying-power deployment {deployed * 100:.0f}% is at/over the "
                    f"{settings.execution.max_bp_usage_pct * 100:.0f}% auto-mode cap"
                )
    needed = preview.init_margin_change
    if available is None:
        # IBK-122 fail-closed: never place blind. If IBKR gives no init-margin
        # or no available-funds figure, reject rather than skip the money gate.
        return _reject(
            "margin preview incomplete (init-margin or available funds unknown) "
            "— refusing to place blind"
        )
    structural_margin_fallback = False
    if needed is None:
        if settings.execution.allow_structural_margin_fallback:
            needed = max_loss_unit * quantity
            structural_margin_fallback = True
        else:
            return _reject(
                "margin preview incomplete (init-margin or available funds unknown) "
                "— refusing to place blind"
            )
    if not math.isfinite(needed) or not math.isfinite(available):
        return _reject(
            "margin preview invalid (init-margin or available funds non-finite) "
            "— refusing to place blind"
        )
    if needed > available:
        return _reject(
            f"margin Δ ${needed:,.0f} exceeds available funds ${available:,.0f}"
        )
    margin_note = f"margin Δ ${needed:,.0f}"
    if structural_margin_fallback:
        margin_note += " (paper fallback: full structural max loss)"
    if preview.warning:
        margin_note += f" ({preview.warning})"

    # 10. Stage (+ decision journal) -> submitting -> place -> submitted.
    nbbo = combo_bid_ask(legs, quotes)
    increment = price_increment_for(symbol)
    budget = (
        slippage_budget(
            nbbo[0], nbbo[1],
            frac=settings.execution.max_slippage_spread_frac,
            abs_cap=settings.execution.max_slippage_abs,
            increment=increment,
        )
        if nbbo is not None
        else increment
    )
    try:
        record = stage_order(engine, score_id, quantity=quantity, now=ts_now)
    except IntegrityError:
        return _reject(
            f"pick {score_id} already has an order intent; its authorization is consumed"
        )
    record_order_quotes(
        engine, record.id, kind="decision", step=0, ts=ts_now,
        combo_bid=nbbo[0] if nbbo else None,
        combo_ask=nbbo[1] if nbbo else None,
        combo_mid=fresh_net, target_net=fresh_net, limit_price=limit_price,
        legs=[
            {
                "expiry": leg["expiry"], "strike": leg["strike"],
                "right": leg["right"], "side": leg["side"],
                "bid": q.bid, "ask": q.ask, "mid": q.mid, "delayed": q.delayed,
            }
            for leg in option_legs
            if (q := quotes.get((str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))))
        ],
    )
    execution_verdict = can_execute(settings, load_state(engine))
    if not execution_verdict.allowed:
        transition(
            engine,
            record.id,
            "skipped",
            error=f"execution interlock closed before placement: {execution_verdict.reason}",
            now=ts_now,
        )
        return ExecuteOutcome(
            ok=False,
            message=f"❌ execution interlock closed before placement: {execution_verdict.reason}",
            order_id=record.id,
        )
    transition(engine, record.id, "submitting", now=ts_now)
    try:
        placed = await deps.order_client.place_combo_limit(
            symbol,
            legs,
            quantity=quantity,
            limit_price=limit_price,
            order_ref=record.order_ref or f"obot-{record.id}",
        )
    except Exception as exc:  # noqa: BLE001 -- broker outcome may be unknown
        halt_reason = (
            f"entry order #{record.id} broker placement outcome unknown after "
            f"exception: {exc}"
        )
        try:
            set_order_note(engine, record.id, halt_reason)
        except Exception:  # noqa: BLE001 -- kill still takes precedence
            log.exception("failed to annotate uncertain entry order #%s", record.id)
        trip_kill(engine, halt_reason, now=ts_now)
        return ExecuteOutcome(
            ok=False,
            message=(
                f"🛑 HALT: order #{record.id} placement outcome unknown: {exc}. "
                "Reconcile broker and ledger state before re-arming."
            ),
            order_id=record.id,
        )
    try:
        set_order_leg_contracts(engine, record.id, placed.leg_contracts)
        transition(
            engine, record.id, "submitted", ib_order_id=placed.ib_order_id, now=ts_now
        )
    except Exception as exc:  # noqa: BLE001 -- broker order is already real
        # A reconcile or persistence race resolved/prevented this transition
        # while we were awaiting the broker. The order may be live at IBKR;
        # halt even if cancellation appears to succeed because its final state
        # is not yet reconciled.
        halt_reason = (
            f"entry order #{record.id} broker/ledger completion race after "
            f"placement of IB order {placed.ib_order_id}: {exc}"
        )
        trip_kill(engine, halt_reason, now=ts_now)
        log.error("%s — cancelling at broker", halt_reason)
        try:
            await deps.order_client.cancel(placed.ib_order_id)
        except Exception:  # noqa: BLE001 -- persisted kill switch is the backstop
            log.exception("race-cancel failed for order #%s", record.id)
        return ExecuteOutcome(
            ok=False,
            message=(
                f"🛑 HALT: order #{record.id} hit a broker/ledger race after "
                "placement; cancellation requested. Reconcile before re-arming."
            ),
            order_id=record.id,
        )

    # 11. Spawn the price walk (IBK-127). Without walk plumbing the order
    # simply rests at mid until the TTL watcher cancels (v1 behavior).
    walking = ""
    if (
        deps.walk_md is not None
        and deps.walk_tasks is not None
        and settings.execution.walk_max_steps > 0
    ):
        task = asyncio.create_task(
            run_price_walk(
                engine=engine, settings=settings,
                order_client=deps.order_client, md=deps.walk_md,
                symbol=symbol, legs=legs, order_id=record.id,
                ib_order_id=placed.ib_order_id, decision_mid=fresh_net,
                budget=budget, increment=increment,
            )
        )
        deps.walk_tasks.add(task)
        task.add_done_callback(deps.walk_tasks.discard)
        walking = (
            f"\nwalking mid→{fresh_net - budget:+.2f} over "
            f"{settings.execution.walk_max_steps}×{settings.execution.walk_step_seconds}s"
            f" + {settings.execution.walk_final_rest_seconds}s rest"
        )
    log.info(
        "executed pick %s -> order #%s (%s %sx %s @ %+.2f, budget %.2f)",
        score_id, record.id, pick.strategy, quantity, symbol, fresh_net, budget,
    )
    kind = "credit" if fresh_net > 0 else "debit"
    return ExecuteOutcome(
        ok=True,
        message=(
            f"✅ submitted #{record.id}: {symbol} {pick.strategy} {quantity}x "
            f"@ net {kind} {abs(fresh_net):.2f}/unit\n{sizing_note}\n"
            f"{margin_note}{walking}\n"
            f"TTL {settings.execution.order_ttl_minutes}m backstop — /orders to "
            f"track, /cancelorder {record.id} to pull{drift_note}"
        ),
        order_id=record.id,
    )
