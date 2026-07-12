"""Tests for automated per-structure exits (IBK-129)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, func, insert, select, update

from optionsbot.config import Settings
from optionsbot.execution.exits import evaluate_exit
from optionsbot.execution.orders import (
    CloseAlreadyClaimed,
    get_order,
    open_close_for,
    record_fill,
    stage_close_order,
)
from optionsbot.storage.schema import orders

NOW = datetime(2026, 6, 11, 16, 0, tzinfo=UTC)

LEGS = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
]


def _settings(**kwargs: object) -> Settings:
    s = Settings()
    for key, value in kwargs.items():
        if hasattr(s.manage, key):
            setattr(s.manage, key, value)
        else:
            setattr(s.execution, key, value)
    return s


# --- evaluate_exit matrix -----------------------------------------------------------


def test_credit_take_profit_at_half_kept() -> None:
    # Collected 1.20; reopening now costs 0.60 -> kept 50% -> TP.
    reason = evaluate_exit(
        entry_net=1.20, current_net=0.60, dte=30, settings=_settings(),
    )
    assert reason is not None and "profit" in reason


def test_credit_below_target_holds() -> None:
    reason = evaluate_exit(
        entry_net=1.20, current_net=0.80, dte=30, settings=_settings(),
    )
    assert reason is None


def test_debit_take_profit_at_half_gain() -> None:
    # Paid 2.00; structure now worth 3.00 -> +50% on debit -> TP.
    reason = evaluate_exit(
        entry_net=-2.00, current_net=-3.00, dte=30, settings=_settings(),
    )
    assert reason is not None and "profit" in reason


def test_soft_stop_disabled_by_default() -> None:
    # Credit 1.20 now costs 3.80 to close: loss 2.60 > 2x credit — but the
    # defined-risk width is the stop; soft stop is off by default.
    reason = evaluate_exit(
        entry_net=1.20, current_net=3.80, dte=30, settings=_settings(),
    )
    assert reason is None


def test_soft_stop_fires_when_enabled() -> None:
    settings = _settings(exit_stop_enabled=True)
    reason = evaluate_exit(entry_net=1.20, current_net=3.80, dte=30, settings=settings)
    assert reason is not None and "stop" in reason


def test_time_exit_at_manage_dte() -> None:
    reason = evaluate_exit(
        entry_net=1.20, current_net=1.10, dte=21, settings=_settings(),
    )
    assert reason is not None and "DTE" in reason


def test_expiry_guard_overrides_everything() -> None:
    reason = evaluate_exit(
        entry_net=1.20, current_net=1.10, dte=3, settings=_settings(),
    )
    assert reason is not None and "expiry" in reason


def test_current_net_none_holds() -> None:
    assert evaluate_exit(entry_net=1.20, current_net=None, dte=30, settings=_settings()) is None


# --- stage_close_order ----------------------------------------------------------------


def _filled_entry(engine: Engine, *, quantity: int = 2) -> int:
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=LEGS, quantity=quantity, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, ib_order_id=11, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(order_ref=f"obot-{order_id}"))
    record_fill(engine, order_id, exec_id=f"e{order_id}a", side="SELL",
                price=1.60, qty=quantity, ts=NOW)
    record_fill(engine, order_id, exec_id=f"e{order_id}b", side="BUY",
                price=0.40, qty=quantity, ts=NOW)
    return order_id


def test_stage_close_order_flips_legs_and_links(tmp_db: Engine) -> None:
    entry_id = _filled_entry(tmp_db)
    entry = get_order(tmp_db, entry_id)
    assert entry is not None
    close = stage_close_order(tmp_db, entry, now=NOW)
    assert close.intent == "close"
    assert close.status == "staged"
    assert close.quantity == entry.quantity
    assert close.order_ref == f"obot-{close.id}"
    assert [leg["side"] for leg in close.legs] == ["buy", "sell"]  # flipped
    assert [leg["strike"] for leg in close.legs] == [580.0, 575.0]
    with tmp_db.connect() as conn:
        from sqlalchemy import select

        row = conn.execute(select(orders).where(orders.c.id == close.id)).one()
    assert row.closes_order_id == entry_id


def test_open_close_for_detects_active_close(tmp_db: Engine) -> None:
    entry_id = _filled_entry(tmp_db)
    entry = get_order(tmp_db, entry_id)
    assert entry is not None
    assert open_close_for(tmp_db, entry_id) is None
    close = stage_close_order(tmp_db, entry, now=NOW)
    found = open_close_for(tmp_db, entry_id)
    assert found is not None and found.id == close.id
    # A failed close stops blocking (retry next tick).
    with tmp_db.begin() as conn:
        conn.execute(update(orders).where(orders.c.id == close.id)
                     .values(status="abandoned", terminal_ts=NOW))
    assert open_close_for(tmp_db, entry_id) is None


def test_stage_close_order_claim_is_atomic(tmp_db: Engine) -> None:
    entry_id = _filled_entry(tmp_db)
    entry = get_order(tmp_db, entry_id)
    assert entry is not None

    first = stage_close_order(tmp_db, entry, now=NOW)
    with pytest.raises(CloseAlreadyClaimed, match=str(first.id)):
        stage_close_order(tmp_db, entry, now=NOW)

    with tmp_db.connect() as conn:
        active_count = conn.execute(
            select(func.count())
            .select_from(orders)
            .where(orders.c.closes_order_id == entry_id)
            .where(orders.c.status.in_(["staged", "submitting", "submitted", "partial"]))
        ).scalar_one()
    assert active_count == 1
