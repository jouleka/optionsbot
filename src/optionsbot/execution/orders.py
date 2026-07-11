"""Order ledger + state machine (IBK-124).

One row per order INTENT. Rows are staged before any network call (so a crash
mid-submit is recoverable by reconciling `submitting` rows against the broker,
IBK-128) and move through an enforced state machine::

    staged ──▶ submitting ──▶ submitted ──▶ (partial ──▶) filled
       │            │              │             │
       │            ├─▶ rejected   ├─▶ cancelled├─▶ cancelled
       └─▶ skipped  └─▶ skipped    ├─▶ rejected └─▶ abandoned
                                   └─▶ abandoned

`skipped` = entry gates failed before/at submit; `abandoned` = price-walk
exhausted and we cancelled by policy (distinct from other cancels so the
track record can tell "no fill at our price" from "we changed our mind").
Illegal transitions raise rather than silently corrupting the ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, Row, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from optionsbot.execution.close_safety import NonAtomicCloseError, assert_atomic_close_legs
from optionsbot.storage.schema import (
    entry_intent_consumptions,
    fills,
    order_quotes,
    orders,
    snapshots,
    strategy_scores,
    walk_state,
)

ORDER_STATUSES: frozenset[str] = frozenset(
    {
        "staged",
        "submitting",
        "submitted",
        "partial",
        "filled",
        "cancelled",
        "rejected",
        "abandoned",
        "skipped",
    }
)

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"filled", "cancelled", "rejected", "abandoned", "skipped"}
)

# Failed terminals: a fill landing on one of these means the broker holds a
# position the ledger denies — the mismatch that must trip the kill switch
# (checked both live in the tracker and during reconciliation replay).
FAILED_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"skipped", "rejected", "cancelled", "abandoned"}
)

# Orders resting at IBKR (have an ib_order_id and may still fill).
WORKING_STATUSES: frozenset[str] = frozenset({"submitted", "partial"})

# Self-loops on the working states are legal NO-OPS: ib_async re-delivers
# orderStatus with an unchanged status string while filled/remaining mutate,
# and that must never raise mid-pipeline. partial -> rejected covers IBKR
# rejecting/deactivating the REMAINDER of a partially-filled order.
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "staged": frozenset({"submitting", "skipped"}),
    "submitting": frozenset({"submitted", "rejected", "skipped"}),
    "submitted": frozenset(
        {"submitted", "partial", "filled", "cancelled", "rejected", "abandoned"}
    ),
    "partial": frozenset({"partial", "filled", "cancelled", "rejected", "abandoned"}),
    "filled": frozenset(),
    "cancelled": frozenset(),
    "rejected": frozenset(),
    "abandoned": frozenset(),
    "skipped": frozenset(),
}

_OPTION_MULTIPLIER = 100


class IllegalOrderTransition(RuntimeError):
    """Raised when a status change violates LEGAL_TRANSITIONS."""


class RealizedPnLUnavailable(RuntimeError):
    """Raised when a filled round trip lacks complete, finite accounting."""


class CloseAlreadyClaimed(RuntimeError):
    """Raised when another active close already owns an entry."""

    def __init__(self, entry_id: int, close_id: int | None) -> None:
        self.entry_id = entry_id
        self.close_id = close_id
        winner = f"close order {close_id}" if close_id is not None else "another close order"
        super().__init__(f"entry {entry_id} is already claimed by {winner}")


@dataclass(frozen=True, slots=True)
class OrderRecord:
    id: int
    strategy_score_id: int | None
    closes_order_id: int | None
    intent: str
    symbol: str
    strategy: str
    legs: list[dict[str, Any]]
    quantity: int
    limit_price: float | None
    ib_order_id: int | None
    ib_perm_id: int | None
    order_ref: str | None
    status: str
    staged_ts: datetime | None
    submitted_ts: datetime | None
    terminal_ts: datetime | None
    last_error: str | None
    reprice_count: int


def _aware(ts: datetime | None) -> datetime | None:
    # SQLite DateTime(timezone=True) drops tzinfo on read; values are written
    # as UTC, so re-attach it (same defense as alert_dedup / execution.state).
    if ts is not None and ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def _to_record(row: Row[Any]) -> OrderRecord:
    return OrderRecord(
        id=row.id,
        strategy_score_id=row.strategy_score_id,
        closes_order_id=row.closes_order_id,
        intent=row.intent,
        symbol=row.symbol,
        strategy=row.strategy,
        legs=list(row.legs_json or []),
        quantity=row.quantity,
        limit_price=row.limit_price,
        ib_order_id=row.ib_order_id,
        ib_perm_id=row.ib_perm_id,
        order_ref=row.order_ref,
        status=row.status,
        staged_ts=_aware(row.staged_ts),
        submitted_ts=_aware(row.submitted_ts),
        terminal_ts=_aware(row.terminal_ts),
        last_error=row.last_error,
        reprice_count=row.reprice_count,
    )


def stage_order(
    engine: Engine,
    strategy_score_id: int,
    *,
    intent: str = "open",
    quantity: int | None = None,
    now: datetime | None = None,
) -> OrderRecord:
    """Stage an order intent from a persisted pick.

    Rebuilds everything from the strategy_scores row (legs_json +
    suggestion_json) and its snapshot's symbol — the same
    reconstruct-from-the-row pattern the alert retry path and the outcomes
    ledger already rely on. ``quantity`` defaults to the pick's
    suggested_quantity; the limit price is deliberately NOT set here (pricing
    happens at submit time from fresh quotes, IBK-126/127).
    """
    ts = now if now is not None else datetime.now(UTC)
    with engine.begin() as conn:
        row = conn.execute(
            select(
                strategy_scores.c.id,
                strategy_scores.c.strategy,
                strategy_scores.c.legs_json,
                strategy_scores.c.suggestion_json,
                snapshots.c.symbol,
            )
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(strategy_scores.c.id == strategy_score_id)
        ).first()
        if row is None:
            raise ValueError(f"no strategy_scores row with id {strategy_score_id}")
        suggestion: dict[str, Any] = row.suggestion_json or {}
        qty = quantity if quantity is not None else int(suggestion.get("suggested_quantity") or 0)
        if qty < 1:
            raise ValueError(f"order quantity must be >= 1, got {qty}")
        inserted = conn.execute(
            insert(orders).values(
                strategy_score_id=row.id,
                intent=intent,
                symbol=row.symbol,
                strategy=row.strategy,
                legs_json=row.legs_json,
                quantity=qty,
                status="staged",
                staged_ts=ts,
                reprice_count=0,
            )
        ).inserted_primary_key
        assert inserted is not None  # single-row INSERT always returns a PK
        order_id = int(inserted[0])
        if intent == "open":
            conn.execute(
                insert(entry_intent_consumptions).values(
                    strategy_score_id=row.id,
                    first_order_id=order_id,
                    consumed_at=ts,
                )
            )
        # Deterministic broker-side tag: stamped into Order.orderRef at submit
        # so reconciliation (IBK-128) can map IBKR orders back to rows.
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(order_ref=f"obot-{order_id}")
        )
    record = get_order(engine, int(order_id))
    assert record is not None  # just inserted in a committed transaction
    return record


def stage_close_order(
    engine: Engine, entry: OrderRecord, *, now: datetime | None = None
) -> OrderRecord:
    """Stage the closing order for a filled entry: every leg side flipped,
    same symbol/strategy/quantity, linked via closes_order_id (IBK-129)."""
    ts = now if now is not None else datetime.now(UTC)
    flipped = [
        {**leg, "side": "buy" if leg.get("side") == "sell" else "sell"}
        for leg in entry.legs
    ]
    try:
        with engine.begin() as conn:
            inserted = conn.execute(
                insert(orders).values(
                    strategy_score_id=entry.strategy_score_id,
                    closes_order_id=entry.id,
                    intent="close",
                    symbol=entry.symbol,
                    strategy=entry.strategy,
                    legs_json=flipped,
                    quantity=entry.quantity,
                    status="staged",
                    staged_ts=ts,
                    reprice_count=0,
                )
            ).inserted_primary_key
            assert inserted is not None
            order_id = int(inserted[0])
            conn.execute(
                update(orders)
                .where(orders.c.id == order_id)
                .values(order_ref=f"obot-{order_id}")
            )
    except IntegrityError as exc:
        winner = open_close_for(engine, entry.id)
        if winner is None:
            raise
        raise CloseAlreadyClaimed(entry.id, winner.id) from exc
    record = get_order(engine, order_id)
    assert record is not None
    return record


def open_close_for(engine: Engine, entry_id: int) -> OrderRecord | None:
    """The ACTIVE (non-terminal) close working against this entry, if any."""
    active = sorted(ORDER_STATUSES - TERMINAL_STATUSES)
    with engine.connect() as conn:
        row = conn.execute(
            select(orders)
            .where(orders.c.closes_order_id == entry_id)
            .where(orders.c.status.in_(active))
            .limit(1)
        ).first()
    return None if row is None else _to_record(row)


def get_order(engine: Engine, order_id: int) -> OrderRecord | None:
    with engine.connect() as conn:
        row = conn.execute(select(orders).where(orders.c.id == order_id)).first()
    return None if row is None else _to_record(row)


def set_order_leg_contracts(
    engine: Engine,
    order_id: int,
    leg_contracts: tuple[tuple[int, int, str], ...],
) -> OrderRecord:
    """Bind exact qualified IBKR contract identities to a staged order."""
    with engine.begin() as conn:
        row = conn.execute(
            select(orders.c.legs_json).where(orders.c.id == order_id)
        ).first()
        if row is None:
            raise ValueError(f"no order with id {order_id}")
        legs = list(row.legs_json or [])
        option_indexes = [
            index
            for index, leg in enumerate(legs)
            if leg.get("sec_type", "OPT") == "OPT"
        ]
        if len(option_indexes) != len(leg_contracts) or not option_indexes:
            raise ValueError("qualified option contract count does not match order legs")
        seen: set[int] = set()
        for index, raw_terms in zip(option_indexes, leg_contracts, strict=True):
            try:
                raw_con_id, raw_multiplier, raw_currency = raw_terms
                con_id = int(raw_con_id)
                multiplier = int(raw_multiplier)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("malformed qualified option contract terms") from exc
            if (
                isinstance(raw_con_id, bool)
                or con_id <= 0
                or con_id != raw_con_id
                or con_id in seen
                or isinstance(raw_multiplier, bool)
                or multiplier != 100
                or multiplier != raw_multiplier
                or raw_currency != "USD"
            ):
                raise ValueError("unsupported or duplicate qualified option contract terms")
            seen.add(con_id)
            legs[index] = {
                **legs[index],
                "con_id": con_id,
                "multiplier": multiplier,
                "currency": raw_currency,
            }
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(legs_json=legs)
        )
    record = get_order(engine, order_id)
    assert record is not None
    return record


def transition(
    engine: Engine,
    order_id: int,
    new_status: str,
    *,
    ib_order_id: int | None = None,
    ib_perm_id: int | None = None,
    error: str | None = None,
    now: datetime | None = None,
) -> OrderRecord:
    """Move an order to ``new_status``, enforcing LEGAL_TRANSITIONS.

    Sets submitted_ts on entering ``submitted``, terminal_ts on entering any
    terminal state, and records ib ids / last_error when provided.
    """
    if new_status not in ORDER_STATUSES:
        raise IllegalOrderTransition(f"unknown status {new_status!r}")
    ts = now if now is not None else datetime.now(UTC)
    # Read-then-update in one transaction: safe under the single-process
    # asyncio daemon (SQLite serializes writers); revisit if a second writer
    # process is ever introduced.
    with engine.begin() as conn:
        row = conn.execute(
            select(orders.c.status).where(orders.c.id == order_id)
        ).first()
        if row is None:
            raise ValueError(f"no order with id {order_id}")
        current: str = row.status
        if new_status not in LEGAL_TRANSITIONS[current]:
            raise IllegalOrderTransition(
                f"order {order_id}: illegal transition {current!r} -> {new_status!r}"
            )
        if new_status != current:
            values: dict[str, Any] = {"status": new_status}
            if ib_order_id is not None:
                values["ib_order_id"] = ib_order_id
            if ib_perm_id is not None:
                values["ib_perm_id"] = ib_perm_id
            if error is not None:
                values["last_error"] = error
            if new_status == "submitted":
                values["submitted_ts"] = ts
            if new_status in TERMINAL_STATUSES:
                values["terminal_ts"] = ts
            conn.execute(update(orders).where(orders.c.id == order_id).values(**values))
        # else: same-status re-delivery — idempotent no-op, nothing rewritten.
    record = get_order(engine, order_id)
    assert record is not None
    return record


def bump_reprice(
    engine: Engine,
    order_id: int,
    *,
    new_limit_price: float,
    now: datetime | None = None,
) -> OrderRecord:
    """Record a price-walk step: only working (submitted/partial) orders reprice."""
    with engine.begin() as conn:
        row = conn.execute(
            select(orders.c.status, orders.c.reprice_count).where(orders.c.id == order_id)
        ).first()
        if row is None:
            raise ValueError(f"no order with id {order_id}")
        if row.status not in WORKING_STATUSES:
            raise IllegalOrderTransition(
                f"order {order_id}: cannot reprice in status {row.status!r}"
            )
        conn.execute(
            update(orders)
            .where(orders.c.id == order_id)
            .values(limit_price=new_limit_price, reprice_count=row.reprice_count + 1)
        )
    record = get_order(engine, order_id)
    assert record is not None
    return record


def record_fill(
    engine: Engine,
    order_id: int,
    *,
    exec_id: str,
    side: str,
    price: float,
    qty: int,
    ts: datetime,
    leg_con_id: int | None = None,
) -> bool:
    """Persist one per-leg execution. Returns False on a duplicate execId
    (IBKR re-sends executions on reconnect — replay must be idempotent).

    ``side`` is an IBKR execution side, uppercase — NOT a legs_json side
    (those are lowercase 'buy'/'sell'); the explicit check turns a wiring
    mistake into a clear error instead of a CHECK-constraint IntegrityError.
    """
    if side not in ("BUY", "SELL"):
        raise ValueError(f"fill side must be 'BUY' or 'SELL', got {side!r}")
    with engine.begin() as conn:
        exists = conn.execute(
            select(fills.c.id).where(fills.c.ib_exec_id == exec_id)
        ).first()
        if exists is not None:
            return False
        conn.execute(
            insert(fills).values(
                order_id=order_id,
                ib_exec_id=exec_id,
                side=side,
                price=price,
                qty=qty,
                ts=ts,
                leg_con_id=leg_con_id,
            )
        )
    return True


def set_fill_commission(engine: Engine, exec_id: str, commission: float) -> bool:
    """Attach a commissionReport amount to its fill (keyed by execId)."""
    with engine.begin() as conn:
        result = conn.execute(
            update(fills).where(fills.c.ib_exec_id == exec_id).values(commission=commission)
        )
    return result.rowcount > 0


def set_order_note(engine: Engine, order_id: int, note: str) -> None:
    """Annotate a row (last_error) WITHOUT changing status — used to record
    intent ("cancel requested: walk exhausted") while the broker still owns
    the order's fate; the tracker performs the actual terminal transition."""
    with engine.begin() as conn:
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(last_error=note)
        )


def record_order_quotes(
    engine: Engine,
    order_id: int,
    *,
    kind: str,
    step: int,
    ts: datetime,
    combo_bid: float | None,
    combo_ask: float | None,
    combo_mid: float | None,
    target_net: float | None,
    limit_price: float | None,
    legs: list[dict[str, Any]],
) -> None:
    """One decision-journal row (IBK-127): the quotes behind a pricing action."""
    with engine.begin() as conn:
        conn.execute(
            insert(order_quotes).values(
                order_id=order_id,
                kind=kind,
                step=step,
                ts=ts,
                combo_bid=combo_bid,
                combo_ask=combo_ask,
                combo_mid=combo_mid,
                target_net=target_net,
                limit_price=limit_price,
                legs_json=legs,
            )
        )


@dataclass(frozen=True, slots=True)
class WalkState:
    """A persisted in-flight price-walk (Work-stream D1)."""

    order_id: int
    ib_order_id: int
    symbol: str
    legs: list[dict[str, Any]]
    decision_mid: float
    budget: float
    increment: float
    step: int
    prev_target: float


def upsert_walk_state(
    engine: Engine,
    order_id: int,
    *,
    ib_order_id: int,
    symbol: str,
    legs: list[dict[str, Any]],
    decision_mid: float,
    budget: float,
    increment: float,
    step: int,
    prev_target: float,
    ts: datetime,
) -> None:
    """Write (or overwrite) the walk-state row for an order. One row per order
    (PK = order_id); the latest step always wins."""
    values = {
        "ib_order_id": ib_order_id,
        "symbol": symbol,
        "legs_json": legs,
        "decision_mid": decision_mid,
        "budget": budget,
        "increment": increment,
        "step": step,
        "prev_target": prev_target,
        "updated_ts": ts,
    }
    with engine.begin() as conn:
        existing = conn.execute(
            select(walk_state.c.order_id).where(walk_state.c.order_id == order_id)
        ).first()
        if existing is None:
            conn.execute(insert(walk_state).values(order_id=order_id, **values))
        else:
            conn.execute(
                update(walk_state).where(walk_state.c.order_id == order_id).values(**values)
            )


def clear_walk_state(engine: Engine, order_id: int) -> None:
    """Delete the walk-state row once a walk ends (fill/cancel/exhaustion)."""
    with engine.begin() as conn:
        conn.execute(delete(walk_state).where(walk_state.c.order_id == order_id))


def load_walk_states(engine: Engine) -> list[WalkState]:
    """All persisted walks whose order is still non-terminal (resume set)."""
    non_terminal = sorted(ORDER_STATUSES - TERMINAL_STATUSES)
    with engine.connect() as conn:
        rows = conn.execute(
            select(walk_state)
            .join(orders, orders.c.id == walk_state.c.order_id)
            .where(orders.c.status.in_(non_terminal))
            .order_by(walk_state.c.order_id)
        ).fetchall()
    return [
        WalkState(
            order_id=r.order_id,
            ib_order_id=r.ib_order_id,
            symbol=r.symbol,
            legs=list(r.legs_json or []),
            decision_mid=r.decision_mid,
            budget=r.budget,
            increment=r.increment,
            step=r.step,
            prev_target=r.prev_target,
        )
        for r in rows
    ]


def net_premium(engine: Engine, order_id: int) -> float | None:
    """Signed premium dollars across this order's fills (None if no fills).

    SELL legs collect, BUY legs pay; per-contract option prices carry the
    standard x100 multiplier. Positive = net credit received.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            select(fills.c.side, fills.c.price, fills.c.qty).where(
                fills.c.order_id == order_id
            )
        ).fetchall()
    if not rows:
        return None
    total = 0.0
    for row in rows:
        sign = 1.0 if row.side == "SELL" else -1.0
        total += sign * row.price * row.qty * _OPTION_MULTIPLIER
    return total


@dataclass(frozen=True, slots=True)
class ClosedPair:
    """One realized round-trip: a filled entry and its filled close (IBK-130/131)."""

    entry_id: int
    close_id: int
    symbol: str
    strategy: str
    quantity: int
    pnl: float  # dollars, commissions included
    closed_ts: datetime | None


def total_commissions(engine: Engine, order_id: int) -> float:
    with engine.connect() as conn:
        rows = conn.execute(
            select(fills.c.commission).where(fills.c.order_id == order_id)
        ).fetchall()
    return sum(float(r.commission) for r in rows if r.commission is not None)


def _complete_order_accounting(engine: Engine, order_id: int) -> tuple[float, float]:
    """Return signed premium and commissions only for a provably complete fill."""
    with engine.connect() as conn:
        order = conn.execute(
            select(orders.c.status, orders.c.quantity, orders.c.legs_json).where(
                orders.c.id == order_id
            )
        ).first()
        fill_rows = conn.execute(
            select(
                fills.c.side,
                fills.c.price,
                fills.c.qty,
                fills.c.commission,
                fills.c.leg_con_id,
            ).where(fills.c.order_id == order_id)
        ).fetchall()
    if order is None or order.status != "filled":
        raise RealizedPnLUnavailable(f"order {order_id}: filled order unavailable")
    quantity = order.quantity
    legs = list(order.legs_json or [])
    if quantity <= 0 or not legs:
        raise RealizedPnLUnavailable(f"order {order_id}: expected fill quantity unavailable")

    expected: dict[int, tuple[str, int]] = {}
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        side = str(leg.get("side", "")).upper()
        raw_ratio = leg.get("quantity", 1)
        raw_con_id = leg.get("con_id")
        try:
            ratio = int(raw_ratio)
            con_id = int(raw_con_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealizedPnLUnavailable(
                f"order {order_id}: expected contract attribution unavailable"
            ) from exc
        if (
            side not in {"BUY", "SELL"}
            or isinstance(raw_ratio, bool)
            or ratio <= 0
            or ratio != raw_ratio
            or isinstance(raw_con_id, bool)
            or con_id <= 0
            or con_id != raw_con_id
            or con_id in expected
        ):
            raise RealizedPnLUnavailable(
                f"order {order_id}: expected contract attribution unavailable"
            )
        expected[con_id] = (side, ratio * quantity)
    if not expected:
        raise RealizedPnLUnavailable(
            f"order {order_id}: expected contract attribution unavailable"
        )

    actual: dict[int, tuple[str, int]] = {}
    premium = 0.0
    commissions = 0.0
    for fill in fill_rows:
        try:
            price = float(fill.price)
            qty = int(fill.qty)
            commission = float(fill.commission)
            con_id = int(fill.leg_con_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealizedPnLUnavailable(f"order {order_id}: fill accounting unavailable") from exc
        expected_leg = expected.get(con_id)
        if (
            expected_leg is None
            or expected_leg[0] != fill.side
            or isinstance(fill.leg_con_id, bool)
            or con_id <= 0
            or con_id != fill.leg_con_id
            or isinstance(fill.qty, bool)
            or qty <= 0
            or qty != fill.qty
            or price < 0
            or not math.isfinite(price)
            or not math.isfinite(commission)
        ):
            raise RealizedPnLUnavailable(
                f"order {order_id}: fill contract attribution unavailable"
            )
        prior = actual.get(con_id, (fill.side, 0))
        actual[con_id] = (fill.side, prior[1] + qty)
        premium += (
            (1.0 if fill.side == "SELL" else -1.0)
            * price
            * qty
            * _OPTION_MULTIPLIER
        )
        commissions += commission
    if actual != expected or not math.isfinite(premium) or not math.isfinite(commissions):
        raise RealizedPnLUnavailable(f"order {order_id}: incomplete fills")
    return premium, commissions


def realized_close_pairs(
    engine: Engine, *, since: datetime | None = None
) -> list[ClosedPair]:
    """All realized round-trips, oldest first. pnl = entry premium + close
    premium − real commissions on both legs (premiums are signed: a credit
    entry collects, its closing buy-back pays)."""
    with engine.connect() as conn:
        query = (
            select(orders)
            .where(orders.c.intent == "close")
            .where(orders.c.status == "filled")
            .where(orders.c.closes_order_id.is_not(None))
            .order_by(orders.c.terminal_ts)
        )
        closes = conn.execute(query).fetchall()
    pairs: list[ClosedPair] = []
    for close in closes:
        closed_ts = _aware(close.terminal_ts)
        if since is not None and (closed_ts is None or closed_ts <= since):
            continue
        entry_premium, entry_commissions = _complete_order_accounting(
            engine, close.closes_order_id
        )
        close_premium, close_commissions = _complete_order_accounting(engine, close.id)
        entry = get_order(engine, int(close.closes_order_id))
        if (
            entry is None
            or entry.status != "filled"
            or entry.symbol != close.symbol
            or entry.quantity != close.quantity
        ):
            raise RealizedPnLUnavailable(
                f"close order {close.id}: exact inverse entry unavailable"
            )
        try:
            assert_atomic_close_legs(
                entry_legs=entry.legs,
                close_legs=list(close.legs_json or []),
            )
        except (NonAtomicCloseError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RealizedPnLUnavailable(
                f"close order {close.id}: close is not the exact inverse of entry "
                f"{entry.id}"
            ) from exc
        commissions = entry_commissions + close_commissions
        pairs.append(
            ClosedPair(
                entry_id=close.closes_order_id,
                close_id=close.id,
                symbol=close.symbol,
                strategy=close.strategy,
                quantity=close.quantity,
                pnl=entry_premium + close_premium - commissions,
                closed_ts=closed_ts,
            )
        )
    return pairs


def open_orders(engine: Engine) -> list[OrderRecord]:
    """All non-terminal orders (staged/submitting/submitted/partial)."""
    non_terminal = sorted(ORDER_STATUSES - TERMINAL_STATUSES)
    with engine.connect() as conn:
        rows = conn.execute(
            select(orders).where(orders.c.status.in_(non_terminal)).order_by(orders.c.id)
        ).fetchall()
    return [_to_record(r) for r in rows]


def open_position_exposure(
    engine: Engine,
) -> dict[int, tuple[tuple[str, str, float, str], int]] | None:
    """Reconstruct exact signed option exposure from every persisted execution.

    Returns ``None`` when contract attribution or fill evidence is incomplete;
    reconciliation must then halt rather than claim broker/ledger agreement.
    """
    with engine.connect() as conn:
        order_rows = conn.execute(
            select(orders.c.id, orders.c.status, orders.c.legs_json).where(
                orders.c.intent.in_(("open", "close"))
            )
        ).fetchall()
        fill_rows = conn.execute(
            select(
                fills.c.order_id,
                fills.c.side,
                fills.c.qty,
                fills.c.leg_con_id,
            )
        ).fetchall()

    orders_by_id = {int(row.id): row for row in order_rows}
    fills_by_order: dict[int, list[Any]] = {}
    for fill in fill_rows:
        fills_by_order.setdefault(int(fill.order_id), []).append(fill)

    for row in order_rows:
        option_legs = [
            leg for leg in (row.legs_json or [])
            if leg.get("sec_type", "OPT") == "OPT"
        ]
        if (
            option_legs
            and row.status in {"partial", "filled"}
            and not fills_by_order.get(int(row.id))
        ):
            return None

    exposure: dict[int, tuple[tuple[str, str, float, str], int]] = {}
    for order_id, order_fills in fills_by_order.items():
        order_row = orders_by_id.get(order_id)
        if order_row is None:
            return None
        leg_specs: dict[int, tuple[str, str, float, str]] = {}
        for leg in order_row.legs_json or []:
            if leg.get("sec_type", "OPT") != "OPT":
                continue
            try:
                raw_con_id = leg["con_id"]
                con_id = int(raw_con_id)
                spec = (
                    str(leg["symbol"]),
                    str(leg["expiry"]),
                    float(leg["strike"]),
                    str(leg["right"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if type(raw_con_id) is not int or con_id <= 0 or con_id in leg_specs:
                return None
            leg_specs[con_id] = spec
        for fill in order_fills:
            try:
                raw_con_id = fill.leg_con_id
                raw_qty = fill.qty
                con_id = int(raw_con_id)
                qty = int(raw_qty)
            except (TypeError, ValueError, OverflowError):
                return None
            side = str(fill.side).upper()
            fill_spec = leg_specs.get(con_id)
            if (
                type(raw_con_id) is not int
                or con_id != raw_con_id
                or type(raw_qty) is not int
                or qty != raw_qty
                or fill_spec is None
                or qty <= 0
                or side not in {"BUY", "SELL"}
            ):
                return None
            signed_qty = qty if side == "BUY" else -qty
            prior = exposure.get(con_id)
            if prior is not None and prior[0] != fill_spec:
                return None
            exposure[con_id] = (
                fill_spec,
                (prior[1] if prior else 0) + signed_qty,
            )

    return {con_id: value for con_id, value in exposure.items() if value[1] != 0}


def working_orders(engine: Engine) -> list[OrderRecord]:
    """Orders resting at IBKR (submitted/partial) — the reprice/TTL sweep set."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(orders)
            .where(orders.c.status.in_(sorted(WORKING_STATUSES)))
            .order_by(orders.c.id)
        ).fetchall()
    return [_to_record(r) for r in rows]
