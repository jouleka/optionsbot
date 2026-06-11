"""Tests for full-auto gates, realized pairs, and the execution report (IBK-130/131)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, insert, update

from optionsbot.execution.engine import execute_pick
from optionsbot.execution.orders import (
    realized_close_pairs,
    record_fill,
    record_order_quotes,
    total_commissions,
)
from optionsbot.storage.schema import orders
from optionsbot.validation.execution_report import execution_report
from tests.unit.test_execution.test_engine import NOW as ENGINE_NOW
from tests.unit.test_execution.test_engine import _deps, _insert_pick

NOW = datetime(2026, 6, 11, 16, 0, tzinfo=UTC)


# --- IBK-130 auto-only engine gates -------------------------------------------------


async def test_auto_mode_rejects_earnings_window(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, raw_json={"earnings_in_window": True})
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert not outcome.ok
    assert "earnings" in outcome.message.lower()


async def test_confirm_mode_allows_earnings_window(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, raw_json={"earnings_in_window": True})
    deps = _deps(tmp_db)  # mode=confirm default
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert outcome.ok


async def test_auto_mode_rejects_at_bp_cap(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    # net_liq 100k, available 60k -> 40% deployed >= 30% cap.
    deps = _deps(tmp_db, available_funds=60_000.0, net_liquidation=100_000.0)
    deps.settings.execution.mode = "auto"
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert not outcome.ok
    assert "buying-power" in outcome.message.lower()


async def test_confirm_mode_ignores_bp_cap(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, available_funds=60_000.0, net_liquidation=100_000.0)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert outcome.ok


async def test_closed_round_trip_frees_the_symbol_cap(tmp_db: Engine) -> None:
    # Opus IBK-130 #1: a fully-closed position is a round-trip, not exposure —
    # the symbol must be re-enterable afterwards at max_per_symbol=1.
    _pair(tmp_db)  # SPY entry+close, both filled
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert outcome.ok, outcome.message


async def test_entry_walk_stops_on_kill(tmp_db: Engine) -> None:
    # Opus IBK-130 #3: a kill mid-walk must stop ENTRY walks before they fill.
    from unittest.mock import AsyncMock, MagicMock

    from optionsbot.config import Settings
    from optionsbot.execution.orders import get_order
    from optionsbot.execution.state import trip_kill
    from optionsbot.execution.walk import run_price_walk
    from tests.unit.test_execution.test_walk import LEGS as WALK_LEGS
    from tests.unit.test_execution.test_walk import _md, _walk_order

    order_id = _walk_order(tmp_db)
    trip_kill(tmp_db, "loss limit")
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = AsyncMock()
    settings = Settings()
    settings.execution.walk_step_seconds = 0
    settings.execution.walk_max_steps = 3
    settings.execution.walk_final_rest_seconds = 0
    await run_price_walk(
        engine=tmp_db, settings=settings, order_client=order_client,
        md=_md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)}),
        symbol="SPY", legs=WALK_LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    order_client.cancel.assert_awaited_once_with(11)
    order_client.modify_price.assert_not_awaited()
    assert get_order(tmp_db, order_id).status == "abandoned"  # type: ignore[union-attr]


# --- IBK-131 realized pairs + report --------------------------------------------------


def _pair(
    engine: Engine, *, entry_credit: float = 1.20, close_debit: float = 0.50,
    commission: float = 0.65, closed_ts: datetime = NOW, strategy: str = "bull_put_spread",
) -> tuple[int, int]:
    with engine.begin() as conn:
        epk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy=strategy, legs_json=[],
            quantity=1, status="filled", staged_ts=NOW - timedelta(days=5),
            submitted_ts=NOW - timedelta(days=5), terminal_ts=NOW - timedelta(days=5),
            reprice_count=0,
        )).inserted_primary_key
        assert epk is not None
        entry_id = int(epk[0])
        cpk = conn.execute(insert(orders).values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy=strategy, legs_json=[], quantity=1, status="filled",
            staged_ts=closed_ts, submitted_ts=closed_ts, terminal_ts=closed_ts,
            reprice_count=0,
        )).inserted_primary_key
        assert cpk is not None
        close_id = int(cpk[0])
        for oid, ref in ((entry_id, f"obot-{entry_id}"), (close_id, f"obot-{close_id}")):
            conn.execute(update(orders).where(orders.c.id == oid).values(order_ref=ref))
    record_fill(engine, entry_id, exec_id=f"p{entry_id}", side="SELL",
                price=entry_credit, qty=1, ts=NOW - timedelta(days=5))
    record_fill(engine, close_id, exec_id=f"p{close_id}", side="BUY",
                price=close_debit, qty=1, ts=closed_ts)
    from optionsbot.execution.orders import set_fill_commission

    set_fill_commission(engine, f"p{entry_id}", commission)
    set_fill_commission(engine, f"p{close_id}", commission)
    return entry_id, close_id


def test_realized_close_pairs_math(tmp_db: Engine) -> None:
    entry_id, close_id = _pair(tmp_db)
    [pair] = realized_close_pairs(tmp_db)
    assert pair.entry_id == entry_id and pair.close_id == close_id
    # (1.20 - 0.50) * 100 - 2 x 0.65 commissions = 68.70
    assert pair.pnl == pytest.approx(68.70)
    assert total_commissions(tmp_db, entry_id) == pytest.approx(0.65)


def test_realized_close_pairs_since_filter(tmp_db: Engine) -> None:
    _pair(tmp_db, closed_ts=NOW - timedelta(days=2))
    _pair(tmp_db, closed_ts=NOW)
    assert len(realized_close_pairs(tmp_db)) == 2
    assert len(realized_close_pairs(tmp_db, since=NOW - timedelta(hours=1))) == 1


def test_execution_report_aggregates(tmp_db: Engine) -> None:
    e1, _ = _pair(tmp_db, entry_credit=1.20, close_debit=0.50)   # +68.70
    _pair(tmp_db, entry_credit=0.80, close_debit=1.20, strategy="iron_condor")  # -41.30
    record_order_quotes(
        tmp_db, e1, kind="decision", step=0, ts=NOW, combo_bid=1.10,
        combo_ask=1.30, combo_mid=1.30, target_net=1.30, limit_price=-1.30,
        legs=[],
    )
    report = execution_report(tmp_db)
    assert report.closed == 2
    assert report.wins == 1
    assert report.total_pnl == pytest.approx(68.70 - 41.30)
    assert report.total_commissions == pytest.approx(4 * 0.65)
    assert set(report.by_strategy) == {"bull_put_spread", "iron_condor"}
    # entry e1: decision mid 1.30, realized 1.20/unit -> slippage 0.10 against us.
    assert report.mean_entry_slippage == pytest.approx(0.10)
    assert report.sample_warning  # 2 << 100 closed trades


def test_execution_report_empty(tmp_db: Engine) -> None:
    report = execution_report(tmp_db)
    assert report.closed == 0
    assert report.sample_warning
