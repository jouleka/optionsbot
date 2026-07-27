"""Price-walk engine (IBK-127): pure pricing math + the per-order walk task.

All net prices live in SIGNED CREDIT-POSITIVE space (positive = we receive,
negative = we pay). Walking toward marketable always DECREASES net — a
credit seller accepts less, a debit buyer pays more — so one formula serves
both directions, and the BAG limit sent to IBKR is always ``-net``.

NOT re-exported from ``optionsbot.execution.__init__`` (imports ibkr
submodules; same import-graph reasoning as tracker.py/engine.py).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.execution.orders import (
    WORKING_STATUSES,
    bump_reprice,
    clear_walk_state,
    get_order,
    load_walk_states,
    record_order_quotes,
    set_order_note,
    upsert_walk_state,
)
from optionsbot.execution.state import load_state, trip_kill

if TYPE_CHECKING:
    from optionsbot.ibkr.market_data import MarketDataClient
    from optionsbot.ibkr.orders import OrderClient
    from optionsbot.ibkr.types import OptionQuote

log = logging.getLogger(__name__)

LegSpec = tuple[str, float, str]

# Net-price increments: SPX/SPXW combos trade in nickels; US equity options
# (penny pilot) in pennies. Extend here if another nickel symbol joins the
# universe.
_NICKEL_SYMBOLS = frozenset({"SPX", "SPXW"})


def price_increment_for(symbol: str) -> float:
    return 0.05 if symbol.upper() in _NICKEL_SYMBOLS else 0.01


def round_to_increment(value: float, increment: float) -> float:
    return round(round(value / increment) * increment, 6)


def combo_bid_ask(
    legs: Sequence[Mapping[str, Any]], quotes: Mapping[LegSpec, OptionQuote]
) -> tuple[float, float] | None:
    """Synthetic combo NBBO in signed credit space (bid = worst receive).

    Never trust a combo ticker — IBKR's BAG quote is synthetic anyway, and
    computing it ourselves keeps the journal honest. None unless every
    option leg has both sides.
    """
    bid = 0.0
    ask = 0.0
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
        quote = quotes.get(spec)
        if quote is None or quote.bid is None or quote.ask is None:
            return None
        quote_bid = max(quote.bid, 0.0)
        quote_ask = quote.ask
        if quote_ask < 0 or quote_ask < quote_bid:
            return None
        ratio = int(leg.get("quantity", 1))
        if leg["side"] == "sell":
            bid += quote_bid * ratio
            ask += quote_ask * ratio
        else:
            bid -= quote_ask * ratio
            ask -= quote_bid * ratio
    return bid, ask


def slippage_budget(
    combo_bid: float, combo_ask: float, *, frac: float, abs_cap: float, increment: float
) -> float:
    """Hard per-order slippage budget: min(frac × spread, abs cap), at least
    one increment (a sub-increment budget cannot walk anywhere)."""
    spread = max(combo_ask - combo_bid, 0.0)
    return max(min(frac * spread, abs_cap), increment)


def next_walk_target(
    *,
    decision_mid: float,
    current_mid: float | None,
    prev_target: float,
    step: int,
    max_steps: int,
    budget: float,
    increment: float,
) -> float:
    """Signed net target for step ``step`` (1-based).

    Re-anchors to the CURRENT mid (never walk a stale price), but the hard
    floor anchors to the DECISION mid — the budget is a per-order promise,
    not a moving target. Monotonic vs prev_target so a favorable market move
    never walks the price backward. Rounded to the net increment; the cap is
    honored within half an increment.
    """
    anchor = current_mid if current_mid is not None else decision_mid
    floor = decision_mid - budget
    candidate = anchor - (step / max_steps) * budget
    target = max(candidate, floor)
    target = min(target, prev_target)
    return round_to_increment(target, increment)


def liquidity_issues(
    legs: Sequence[Mapping[str, Any]],
    quotes: Mapping[LegSpec, OptionQuote],
    *,
    leg_spread_frac: float,
    leg_spread_floor: float,
    min_open_interest: int,
) -> list[str]:
    """Per-leg SANITY check — catches broken/garbage quotes, not economics.

    A leg's bid/ask spread must be within max(frac x leg mid, floor $): the
    percentage scales the allowance with the option's price (a $0.55 spread is
    fine on a $14 option, awful on a $0.30 one), and the floor lets cheap,
    genuinely-tight options through. The combo-vs-credit economic gate
    (``combo_spread_issue``) does the real work; OI is checked only when
    enabled AND present (delayed snapshots often omit it).
    """
    issues: list[str] = []
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
        label = f"{spec[0]} {spec[1]:g}{spec[2]}"
        quote = quotes.get(spec)
        if quote is None or quote.bid is None or quote.ask is None:
            issues.append(f"no bid/ask quote on {label}")
            continue
        if quote.ask < quote.bid:
            issues.append(f"crossed quote on {label} ({quote.bid}/{quote.ask})")
            continue
        spread = quote.ask - quote.bid
        mid = (quote.bid + quote.ask) / 2
        allowed = max(leg_spread_frac * mid, leg_spread_floor)
        if spread > allowed:
            issues.append(
                f"spread ${spread:.2f} on {label} exceeds ${allowed:.2f} "
                f"({leg_spread_frac * 100:.0f}% of mid)"
            )
        if (
            min_open_interest > 0
            and quote.open_interest is not None
            and quote.open_interest < min_open_interest
        ):
            issues.append(
                f"open interest {quote.open_interest} on {label} below {min_open_interest}"
            )
    return issues


def combo_spread_issue(
    legs: Sequence[Mapping[str, Any]],
    quotes: Mapping[LegSpec, OptionQuote],
    net_premium: float,
    *,
    max_frac: float,
) -> str | None:
    """The ECONOMIC gate: the combo's bid/ask spread (summed per-leg NBBO)
    must not exceed ``max_frac`` of the net premium being captured — else
    slippage on entry+exit would eat the edge. Returns a reason or None.

    The summed-leg spread overstates the true package spread (a real combo
    quotes tighter), so this is intentionally generous and paired with the
    price-walk, which never pays more than its budget past mid.
    """
    nbbo = combo_bid_ask(legs, quotes)
    if nbbo is None:
        return None  # missing legs already flagged per-leg
    spread = nbbo[1] - nbbo[0]
    juice = abs(net_premium)
    if juice <= 0:
        return None  # zero-premium edge handled upstream
    if spread > max_frac * juice:
        return (
            f"combo bid/ask ${spread:.2f} is {spread / juice * 100:.0f}% of the "
            f"${juice:.2f} net premium (cap {max_frac * 100:.0f}%) — slippage "
            "would eat the edge"
        )
    return None


async def run_price_walk(
    *,
    engine: Engine,
    settings: Settings,
    order_client: OrderClient,
    md: MarketDataClient,
    symbol: str,
    legs: Sequence[Mapping[str, Any]],
    order_id: int,
    ib_order_id: int,
    decision_mid: float,
    budget: float,
    increment: float,
    ibkr_lock: asyncio.Lock | None = None,
    start_step: int = 0,
    prev_target_override: float | None = None,
) -> None:
    """Walk one working order from mid toward marketable, then give up.

    Runs as a fire-and-forget task. When the quote client shares the daemon's
    single market-data session, ``ibkr_lock`` serializes the complete leg set
    against scans so IBKR never sees competing API quote streams. Never raises;
    the TTL watcher remains the backstop if this task dies.
    """
    # Local import to avoid a module-level execution->engine cycle.
    from optionsbot.execution.engine import combo_mid

    cfg = settings.execution
    prev_target = prev_target_override if prev_target_override is not None else decision_mid
    option_legs = [leg for leg in legs if leg.get("sec_type", "OPT") == "OPT"]
    broker_mutated = False
    try:
        for step in range(start_step + 1, cfg.walk_max_steps + 1):
            if cfg.walk_step_seconds:
                await asyncio.sleep(cfg.walk_step_seconds)
            record = get_order(engine, order_id)
            if record is None or record.status not in WORKING_STATUSES:
                return  # filled/cancelled/rejected while we slept — done
            if record.intent == "open" and load_state(engine).killed:
                # A kill switch tripped mid-walk must stop ENTRY walks before
                # they fill (exits keep walking — we still want out).
                await _request_cancel_and_confirm(
                    engine, order_client, order_id, ib_order_id,
                    note="kill switch tripped mid-walk — cancel requested",
                )
                return

            quotes: dict[LegSpec, OptionQuote] = {}
            current_mid: float | None = None
            nbbo: tuple[float, float] | None = None
            try:
                quote_guard = ibkr_lock if ibkr_lock is not None else nullcontext()
                async with quote_guard:
                    for leg in option_legs:
                        spec = (
                            str(leg["expiry"]),
                            float(leg["strike"]),
                            str(leg["right"]),
                        )
                        quotes[spec] = await md.get_option_snapshot(
                            symbol, spec[0], spec[1], spec[2]  # type: ignore[arg-type]
                        )
                if any(quote.delayed is not False for quote in quotes.values()):
                    log.warning(
                        "walk %s step %d: delayed or unknown quote delivery; not repricing",
                        order_id,
                        step,
                    )
                    continue
                if any(not isinstance(quote.ts, datetime) for quote in quotes.values()):
                    log.warning(
                        "walk %s step %d: quote timestamp unknown; not repricing",
                        order_id,
                        step,
                    )
                    continue
                quote_now = datetime.now(UTC)
                quote_max_age = timedelta(seconds=cfg.exit_quote_max_age_seconds)
                invalid_age = False
                for quote in quotes.values():
                    quote_ts = quote.ts
                    if not isinstance(quote_ts, datetime):  # narrowed above; mypy guard
                        invalid_age = True
                        break
                    if quote_ts.tzinfo is None:
                        quote_ts = quote_ts.replace(tzinfo=UTC)
                    quote_age = quote_now - quote_ts.astimezone(UTC)
                    if quote_age < timedelta(0) or quote_age > quote_max_age:
                        invalid_age = True
                        break
                if invalid_age:
                    log.warning(
                        "walk %s step %d: quote age outside exit execution window; "
                        "not repricing",
                        order_id,
                        step,
                    )
                    continue
                current_mid = combo_mid(legs, quotes)
                nbbo = combo_bid_ask(legs, quotes)
                if current_mid is None or nbbo is None:
                    log.warning(
                        "walk %s step %d: incomplete quote set; not repricing",
                        order_id,
                        step,
                    )
                    continue
            except Exception:  # noqa: BLE001 -- fail closed; the next step retries refresh
                log.exception("walk %s step %d: quote refresh failed", order_id, step)
                continue

            if record.intent == "open" and load_state(engine).killed:
                await _request_cancel_and_confirm(
                    engine,
                    order_client,
                    order_id,
                    ib_order_id,
                    note="kill switch tripped during quote refresh — cancel requested",
                )
                return

            target = next_walk_target(
                decision_mid=decision_mid,
                current_mid=current_mid,
                prev_target=prev_target,
                step=step,
                max_steps=cfg.walk_max_steps,
                budget=budget,
                increment=increment,
            )
            if abs(target - prev_target) < increment / 2:
                continue  # no-op step (favorable market or floor reached)
            try:
                await order_client.modify_price(ib_order_id, new_limit_price=-target)
                broker_mutated = True
            except Exception as exc:  # noqa: BLE001 -- broker outcome is unknown
                halt_reason = (
                    f"price-walk modify outcome unknown for order #{order_id}: {exc}"
                )
                try:
                    trip_kill(engine, halt_reason)
                except Exception:  # noqa: BLE001 -- cancellation still mandatory
                    log.exception("walk %s: failed to persist kill switch", order_id)
                log.exception("walk %s step %d: modify outcome unknown", order_id, step)
                await _request_cancel_and_confirm(
                    engine,
                    order_client,
                    order_id,
                    ib_order_id,
                    note=f"{halt_reason}; cancel requested",
                )
                return
            try:
                post_modify_record = get_order(engine, order_id)
                if post_modify_record is None:
                    raise RuntimeError("order row unavailable after broker modify")
                if post_modify_record.status == "filled":
                    return
                if post_modify_record.status not in WORKING_STATUSES:
                    raise RuntimeError(
                        "order entered an unconfirmed non-working state after broker modify: "
                        f"{post_modify_record.status}"
                    )
                bump_reprice(engine, order_id, new_limit_price=-target)
                prev_target = target
                leg_rows = []
                for leg in option_legs:
                    spec = (
                        str(leg["expiry"]),
                        float(leg["strike"]),
                        str(leg["right"]),
                    )
                    q = quotes.get(spec)
                    if q is None:
                        continue
                    leg_rows.append(
                        {
                            "expiry": leg["expiry"],
                            "strike": leg["strike"],
                            "right": leg["right"],
                            "side": leg["side"],
                            "bid": q.bid,
                            "ask": q.ask,
                            "mid": q.mid,
                            "delayed": q.delayed,
                        }
                    )
                record_order_quotes(
                    engine,
                    order_id,
                    kind="step",
                    step=step,
                    ts=datetime.now(UTC),
                    combo_bid=nbbo[0] if nbbo else None,
                    combo_ask=nbbo[1] if nbbo else None,
                    combo_mid=current_mid,
                    target_net=target,
                    limit_price=-target,
                    legs=leg_rows,
                )
                upsert_walk_state(
                    engine,
                    order_id,
                    ib_order_id=ib_order_id,
                    symbol=symbol,
                    legs=[dict(leg) for leg in legs],
                    decision_mid=decision_mid,
                    budget=budget,
                    increment=increment,
                    step=step,
                    prev_target=prev_target,
                    ts=datetime.now(UTC),
                )
            except Exception as exc:  # noqa: BLE001 -- broker already mutated
                halt_reason = (
                    f"price-walk ledger finalization failed after broker modify for "
                    f"order #{order_id}: {exc}"
                )
                try:
                    trip_kill(engine, halt_reason)
                except Exception:  # noqa: BLE001 -- cancellation still mandatory
                    log.exception("walk %s: failed to persist kill switch", order_id)
                log.exception("walk %s step %d: post-modify ledger failure", order_id, step)
                await _request_cancel_and_confirm(
                    engine,
                    order_client,
                    order_id,
                    ib_order_id,
                    note=f"{halt_reason}; cancel requested",
                )
                return

        # Final price rests, then the trade is skipped: request the cancel and
        # let the TRACKER confirm it. Marking the row terminal ourselves while
        # the broker could still (partially) fill is exactly how a real
        # position once hid behind a "trade skipped" message.
        if cfg.walk_final_rest_seconds:
            await asyncio.sleep(cfg.walk_final_rest_seconds)
        record = get_order(engine, order_id)
        if record is None:
            if broker_mutated:
                raise RuntimeError("order row unavailable before final cancellation")
            return
        if record.status not in WORKING_STATUSES:
            if broker_mutated and record.status != "filled":
                raise RuntimeError(
                    "order entered an unconfirmed non-working state before final "
                    f"cancellation: {record.status}"
                )
            return
        await _request_cancel_and_confirm(
            engine, order_client, order_id, ib_order_id,
            note=(
                f"price walk exhausted ({cfg.walk_max_steps} steps + "
                f"{cfg.walk_final_rest_seconds}s rest) — cancel requested"
            ),
        )
    except asyncio.CancelledError:
        raise  # daemon shutdown — let it propagate
    except Exception as exc:  # noqa: BLE001 -- the walk must never crash the daemon
        log.exception("price walk for order %s died", order_id)
        if broker_mutated:
            halt_reason = (
                f"price-walk state became unavailable after broker modification for "
                f"order #{order_id}: {exc}"
            )
            try:
                trip_kill(engine, halt_reason)
            except Exception:  # noqa: BLE001 -- cancellation still mandatory
                log.exception("walk %s: failed to persist kill switch", order_id)
            await _request_cancel_and_confirm(
                engine,
                order_client,
                order_id,
                ib_order_id,
                note=f"{halt_reason}; cancel requested",
            )
    finally:
        # Best-effort: never let cleanup obscure a broker-side uncertainty.
        try:
            record = get_order(engine, order_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("price walk cleanup could not read order %s", order_id)
            if broker_mutated:
                _trip_walk_kill(
                    engine,
                    order_id,
                    f"price-walk cleanup state unavailable after broker modification: {exc}",
                )
        else:
            if record is None:
                if broker_mutated:
                    _trip_walk_kill(
                        engine,
                        order_id,
                        "price-walk cleanup lost order row after broker modification",
                    )
            elif record.status not in WORKING_STATUSES:
                try:
                    clear_walk_state(engine, order_id)
                except Exception as exc:  # noqa: BLE001
                    log.exception("price walk cleanup failed for order %s", order_id)
                    if broker_mutated:
                        _trip_walk_kill(
                            engine,
                            order_id,
                            f"price-walk cleanup persistence failed after broker "
                            f"modification: {exc}",
                        )


_CANCEL_CONFIRM_TIMEOUT = 30.0  # seconds; tests may monkeypatch
_CANCEL_CONFIRM_POLL = 0.5


def _trip_walk_kill(engine: Engine, order_id: int, reason: str) -> None:
    try:
        trip_kill(engine, reason)
    except Exception:  # noqa: BLE001 -- best effort when ledger itself is degraded
        log.exception("walk %s: failed to persist kill switch", order_id)


async def _request_cancel_and_confirm(
    engine: Engine,
    order_client: OrderClient,
    order_id: int,
    ib_order_id: int,
    *,
    note: str,
) -> None:
    """Annotate intent, request the broker cancel, and wait for the TRACKER to
    confirm the terminal state (cancelled — or filled, if a fill won the
    race). Never writes a terminal status itself: the broker owns the order's
    fate until it says otherwise."""
    try:
        set_order_note(engine, order_id, note)
    except Exception as exc:  # noqa: BLE001 -- cancellation remains mandatory
        log.exception("order %s: failed to persist cancel note", order_id)
        _trip_walk_kill(
            engine,
            order_id,
            f"cancel note persistence failed for order #{order_id}: {exc}",
        )
    try:
        await order_client.cancel(ib_order_id)
    except Exception as exc:  # noqa: BLE001 -- broker cancel outcome is unknown
        log.exception("order %s: cancel request outcome unknown", order_id)
        _trip_walk_kill(
            engine,
            order_id,
            f"cancel request outcome unknown for order #{order_id}: {exc}",
        )
        return
    waited = 0.0
    while waited <= _CANCEL_CONFIRM_TIMEOUT:
        try:
            record = get_order(engine, order_id)
        except Exception as exc:  # noqa: BLE001 -- cancel was already requested
            log.exception("order %s: cannot confirm cancellation in ledger", order_id)
            _trip_walk_kill(
                engine,
                order_id,
                f"cancel confirmation unavailable for order #{order_id}: {exc}",
            )
            return
        if record is None:
            _trip_walk_kill(
                engine,
                order_id,
                f"cancel confirmation lost order row #{order_id}",
            )
            return
        if record.status not in WORKING_STATUSES:
            return  # tracker confirmed (cancelled / filled)
        await asyncio.sleep(_CANCEL_CONFIRM_POLL)
        waited += _CANCEL_CONFIRM_POLL
    log.error(
        "order %s: cancel UNCONFIRMED after %.0fs — order may still be working; "
        "TTL watcher keeps retrying", order_id, _CANCEL_CONFIRM_TIMEOUT,
    )
    _trip_walk_kill(
        engine,
        order_id,
        f"cancel unconfirmed for order #{order_id} after "
        f"{_CANCEL_CONFIRM_TIMEOUT:.0f}s",
    )


async def resume_walks(
    *,
    engine: Engine,
    settings: Settings,
    order_client: OrderClient,
    md: MarketDataClient,
    walk_tasks: set[asyncio.Task[None]],
    ibkr_lock: asyncio.Lock | None = None,
    notify: Callable[[str], Awaitable[None]] | None = None,
) -> int:
    """Re-attach persisted price-walks after a restart (Work-stream D1).

    For every non-terminal walk_state row, spawn a run_price_walk that RESUMES
    from the persisted step (never replaying earlier repricings). The adopted
    order is unmanaged until this fires, so we alert per resumed walk. Returns
    the number of walks resumed.
    """
    resumed = 0
    for ws in load_walk_states(engine):
        record = get_order(engine, ws.order_id)
        if record is None or record.status not in WORKING_STATUSES:
            clear_walk_state(engine, ws.order_id)
            continue
        if notify is not None:
            try:
                await notify(
                    f"↻ re-attached price walk for order #{ws.order_id} "
                    f"({ws.symbol}) — resuming from step {ws.step}; it was "
                    "unmanaged since the restart"
                )
            except Exception:  # noqa: BLE001 -- notify failure must not block resume
                log.exception("resume_walks: notify failed for order %s", ws.order_id)
        task = asyncio.create_task(
            run_price_walk(
                engine=engine, settings=settings, order_client=order_client,
                md=md, symbol=ws.symbol, legs=ws.legs, order_id=ws.order_id,
                ib_order_id=ws.ib_order_id, decision_mid=ws.decision_mid,
                budget=ws.budget, increment=ws.increment,
                ibkr_lock=ibkr_lock,
                start_step=ws.step, prev_target_override=ws.prev_target,
            )
        )
        walk_tasks.add(task)
        task.add_done_callback(walk_tasks.discard)
        resumed += 1
    return resumed
