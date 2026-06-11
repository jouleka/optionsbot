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

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, Row, insert, select, update

from optionsbot.storage.schema import (
    fills,
    order_quotes,
    orders,
    snapshots,
    strategy_scores,
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


@dataclass(frozen=True, slots=True)
class OrderRecord:
    id: int
    strategy_score_id: int | None
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
        # Deterministic broker-side tag: stamped into Order.orderRef at submit
        # so reconciliation (IBK-128) can map IBKR orders back to rows.
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(order_ref=f"obot-{order_id}")
        )
    record = get_order(engine, int(order_id))
    assert record is not None  # just inserted in a committed transaction
    return record


def get_order(engine: Engine, order_id: int) -> OrderRecord | None:
    with engine.connect() as conn:
        row = conn.execute(select(orders).where(orders.c.id == order_id)).first()
    return None if row is None else _to_record(row)


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


def open_orders(engine: Engine) -> list[OrderRecord]:
    """All non-terminal orders (staged/submitting/submitted/partial)."""
    non_terminal = sorted(ORDER_STATUSES - TERMINAL_STATUSES)
    with engine.connect() as conn:
        rows = conn.execute(
            select(orders).where(orders.c.status.in_(non_terminal)).order_by(orders.c.id)
        ).fetchall()
    return [_to_record(r) for r in rows]


def working_orders(engine: Engine) -> list[OrderRecord]:
    """Orders resting at IBKR (submitted/partial) — the reprice/TTL sweep set."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(orders)
            .where(orders.c.status.in_(sorted(WORKING_STATUSES)))
            .order_by(orders.c.id)
        ).fetchall()
    return [_to_record(r) for r in rows]
