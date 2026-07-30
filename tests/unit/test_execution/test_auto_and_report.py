"""Tests for full-auto gates, realized pairs, and the execution report (IBK-130/131)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, delete, insert, select, update

from optionsbot.execution.engine import execute_pick
from optionsbot.execution.orders import (
    RealizedPnLUnavailable,
    expiration_assignment_shares,
    get_order,
    open_position_exposure,
    realized_close_pairs,
    record_expiration_settlement,
    record_expired_worthless_settlement,
    record_fill,
    record_order_quotes,
    set_order_leg_contracts,
    settled_entry_ids,
    stage_order,
    total_commissions,
    transition,
)
from optionsbot.ibkr.types import MarginPreview
from optionsbot.storage.schema import fills, orders, position_settlements
from optionsbot.validation.execution_report import execution_report
from tests.unit.test_execution.test_engine import (
    CONDOR_LEGS,
    _deps,
    _insert_pick,
)
from tests.unit.test_execution.test_engine import (
    NOW as ENGINE_NOW,
)

NOW = datetime(2026, 6, 11, 16, 0, tzinfo=UTC)


# --- IBK-130 auto-only engine gates -------------------------------------------------


async def test_auto_mode_rejects_earnings_window(tmp_db: Engine) -> None:
    score_id = _insert_pick(
        tmp_db,
        raw_json={
            "delayed": False,
            "warming_up": False,
            "earnings_in_window": True,
        },
    )
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert not outcome.ok
    assert "earnings" in outcome.message.lower()


async def test_auto_paper_profile_can_trade_defined_risk_through_earnings(
    tmp_db: Engine,
) -> None:
    score_id = _insert_pick(
        tmp_db,
        raw_json={
            "delayed": False,
            "warming_up": False,
            "earnings_in_window": True,
        },
    )
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"
    deps.settings.execution.auto_skip_earnings = False

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok, outcome.message


async def test_paper_profile_uses_structural_max_loss_when_whatif_margin_missing(
    tmp_db: Engine,
) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    deps.settings.execution.allow_structural_margin_fallback = True
    deps.order_client.whatif_combo.return_value = MarginPreview(
        init_margin_change=None,
        maint_margin_change=None,
        equity_with_loan_change=None,
        commission=None,
        max_commission=None,
        warning="paper BAG omitted margin",
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok, outcome.message
    assert "structural max loss" in outcome.message.lower()


async def test_zero_dte_physical_settlement_caps_quantity_before_margin_fallback(
    tmp_db: Engine,
) -> None:
    same_day_legs = [
        {**leg, "expiry": ENGINE_NOW.strftime("%Y%m%d")}
        for leg in CONDOR_LEGS
    ]
    score_id = _insert_pick(
        tmp_db,
        legs=same_day_legs,
        credit_or_debit=450.0,
        max_loss=50.0,
        max_profit=450.0,
    )
    deps = _deps(
        tmp_db,
        available_funds=10_000.0,
        net_liquidation=10_000.0,
        md_mids={(580.0, "P"): 5.0, (575.0, "P"): 0.5},
    )
    deps.settings.execution.zero_dte_only = True
    deps.settings.execution.allow_structural_margin_fallback = True
    deps.order_client.whatif_combo.return_value = MarginPreview(
        init_margin_change=None,
        maint_margin_change=None,
        equity_with_loan_change=None,
        commission=None,
        max_commission=None,
        warning="paper BAG omitted margin",
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok, outcome.message
    placed = deps.order_client.place_combo_limit.await_args
    assert placed.kwargs["quantity"] == 1
    assert "physical-settlement fallback capped" in outcome.message
    assert deps.order_client.whatif_combo.await_count == 2


@pytest.mark.parametrize("quality_flag", ["delayed", "warming_up"])
async def test_auto_mode_rejects_unready_snapshot(
    tmp_db: Engine,
    quality_flag: str,
) -> None:
    raw_json = {"delayed": False, "warming_up": False}
    raw_json[quality_flag] = True
    score_id = _insert_pick(tmp_db, raw_json=raw_json)
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert not outcome.ok
    assert quality_flag.replace("_", " ") in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_auto_mode_accepts_explicit_hv_rank_proxy_during_iv_warmup(
    tmp_db: Engine,
) -> None:
    score_id = _insert_pick(
        tmp_db,
        raw_json={
            "delayed": False,
            "warming_up": True,
            "iv_rank_is_proxy": True,
            "earnings_in_window": False,
        },
    )
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok, outcome.message


async def test_confirm_mode_allows_earnings_window(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, raw_json={"earnings_in_window": True})
    deps = _deps(tmp_db)  # mode=confirm default
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert outcome.ok


async def test_auto_mode_rejects_when_bot_option_heat_reaches_bp_cap(
    tmp_db: Engine,
) -> None:
    existing_score_id = _insert_pick(
        tmp_db,
        symbol="QQQ",
        legs=[{**leg, "symbol": "QQQ"} for leg in CONDOR_LEGS],
        raw_json={"delayed": False, "warming_up": False},
        max_loss=40_000.0,
    )
    existing = stage_order(tmp_db, existing_score_id, quantity=1, now=ENGINE_NOW)
    with tmp_db.begin() as conn:
        conn.execute(
            update(orders)
            .where(orders.c.id == existing.id)
            .values(limit_price=-1.20)
        )
    score_id = _insert_pick(
        tmp_db,
        raw_json={"delayed": False, "warming_up": False},
    )
    deps = _deps(tmp_db, available_funds=60_000.0, net_liquidation=100_000.0)
    deps.settings.execution.mode = "auto"
    deps.settings.execution.max_portfolio_heat_pct = 0.50
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert not outcome.ok
    assert "optionsbot risk deployment" in outcome.message.lower()


async def test_auto_mode_ignores_foreign_margin_in_bp_deployment(
    tmp_db: Engine,
) -> None:
    score_id = _insert_pick(
        tmp_db,
        raw_json={"delayed": False, "warming_up": False},
    )
    # Forty percent of account buying power is consumed outside OptionsBot
    # (for example by a manual MNQ position). The independent affordability
    # gate still sees $60k available, but the bot-owned deployment is zero.
    deps = _deps(tmp_db, available_funds=60_000.0, net_liquidation=100_000.0)
    deps.settings.execution.mode = "auto"

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok, outcome.message


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
    from tests.unit.test_execution.test_walk import _md, _tracker_confirms_cancel, _walk_order

    order_id = _walk_order(tmp_db)
    trip_kill(tmp_db, "loss limit")
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
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
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "cancelled"
    assert "kill switch" in (record.last_error or "")


def test_dynamic_sizing_matrix() -> None:
    # IBK-133 pure math: $5k equity, base 3% = $150.
    from optionsbot.execution.sizing import dynamic_quantity

    # Spread with $90 max loss: budget $150×0.5 (no-edge tilt) = $75 → min-1.
    d = dynamic_quantity(
        equity=5_000, max_loss_unit=90, max_profit_unit=30, prob_profit=0.70,
        open_heat=0, recent_pnls=[], base_risk_pct=0.03, heat_cap_pct=0.15,
        single_trade_cap_pct=0.10,
    )
    assert d.quantity == 1 and "min-1" in d.note

    # Strong edge (b=2, p=0.6 → kelly 0.4 → tilt capped ×2): budget $300 → 5x.
    d = dynamic_quantity(
        equity=5_000, max_loss_unit=60, max_profit_unit=120, prob_profit=0.60,
        open_heat=0, recent_pnls=[], base_risk_pct=0.03, heat_cap_pct=0.15,
        single_trade_cap_pct=0.10,
    )
    assert d.quantity == 5

    # Anti-martingale: 3 straight losses halve the budget.
    d = dynamic_quantity(
        equity=5_000, max_loss_unit=60, max_profit_unit=120, prob_profit=0.60,
        open_heat=0, recent_pnls=[-10, -20, -5], base_risk_pct=0.03,
        heat_cap_pct=0.15, single_trade_cap_pct=0.10,
    )
    assert d.quantity == 2  # 300×0.5 = 150 → 2x

    # $18,885 CSP max loss on $5k: hard single-trade ceiling → 0.
    d = dynamic_quantity(
        equity=5_000, max_loss_unit=18_885, max_profit_unit=615, prob_profit=0.72,
        open_heat=0, recent_pnls=[], base_risk_pct=0.03, heat_cap_pct=0.15,
        single_trade_cap_pct=0.10,
    )
    assert d.quantity == 0 and "single-trade cap" in d.note

    # Heat cap: $700 already open of a $750 cap leaves no room for a $90 trade.
    d = dynamic_quantity(
        equity=5_000, max_loss_unit=90, max_profit_unit=30, prob_profit=0.70,
        open_heat=700, recent_pnls=[], base_risk_pct=0.03, heat_cap_pct=0.15,
        single_trade_cap_pct=0.10,
    )
    assert d.quantity == 0 and "heat" in d.note


def test_open_heat_includes_scoreless_adopted_defined_risk_positions(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    spy_legs = [
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 717.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 717},
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 734.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 734},
    ]
    tlt_legs = [
        {"symbol": "TLT", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 85.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 850},
        {"symbol": "TLT", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 88.0, "right": "C", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 880},
    ]

    with tmp_db.begin() as conn:
        spy_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=spy_legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
        tlt_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="TLT", strategy="long_strangle",
            legs_json=tlt_legs, quantity=6, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, spy_id, exec_id="adopt-spy-long", side="BUY",
                price=5.167983, qty=1, ts=NOW, leg_con_id=717)
    record_fill(tmp_db, spy_id, exec_id="adopt-spy-short", side="SELL",
                price=8.031819, qty=1, ts=NOW, leg_con_id=734)
    record_fill(tmp_db, tlt_id, exec_id="adopt-tlt-put", side="BUY",
                price=0.5941991665, qty=6, ts=NOW, leg_con_id=850)
    record_fill(tmp_db, tlt_id, exec_id="adopt-tlt-call", side="BUY",
                price=0.4841991665, qty=6, ts=NOW, leg_con_id=880)

    expected_spy = (17.0 - 2.863836) * 100
    expected_tlt = (0.5941991665 + 0.4841991665) * 6 * 100
    assert open_heat_dollars(tmp_db) == pytest.approx(expected_spy + expected_tlt)


def test_scored_open_heat_uses_reconstructed_loss_not_forged_suggestion(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    score_id = _insert_pick(tmp_db, max_loss=1.0)
    order = stage_order(tmp_db, score_id, quantity=1, now=NOW)
    set_order_leg_contracts(
        tmp_db,
        order.id,
        ((580001, 100, "USD"), (575001, 100, "USD")),
    )
    transition(tmp_db, order.id, "submitting", now=NOW)
    transition(tmp_db, order.id, "submitted", ib_order_id=77, now=NOW)
    record_fill(
        tmp_db,
        order.id,
        exec_id="scored-short",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=580001,
    )
    record_fill(
        tmp_db,
        order.id,
        exec_id="scored-long",
        side="BUY",
        price=0.40,
        qty=1,
        ts=NOW,
        leg_con_id=575001,
    )
    transition(tmp_db, order.id, "filled", now=NOW)

    assert open_heat_dollars(tmp_db) == pytest.approx(380.0)


def test_open_heat_rejects_fractional_persisted_order_quantity(tmp_db: Engine) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    score_id = _insert_pick(tmp_db, max_loss=380.0)
    order = stage_order(tmp_db, score_id, quantity=1, now=NOW)
    set_order_leg_contracts(
        tmp_db,
        order.id,
        ((580001, 100, "USD"), (575001, 100, "USD")),
    )
    transition(tmp_db, order.id, "submitting", now=NOW)
    transition(tmp_db, order.id, "submitted", ib_order_id=77, now=NOW)
    with tmp_db.begin() as conn:
        conn.execute(
            update(orders).where(orders.c.id == order.id).values(quantity=1.5)
        )

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_rejects_fractional_fill_contract_identity(tmp_db: Engine) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    score_id = _insert_pick(tmp_db, max_loss=380.0)
    order = stage_order(tmp_db, score_id, quantity=1, now=NOW)
    set_order_leg_contracts(
        tmp_db,
        order.id,
        ((580001, 100, "USD"), (575001, 100, "USD")),
    )
    transition(tmp_db, order.id, "submitting", now=NOW)
    transition(tmp_db, order.id, "submitted", ib_order_id=77, now=NOW)
    record_fill(
        tmp_db,
        order.id,
        exec_id="fractional-id-short",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=580001,
    )
    record_fill(
        tmp_db,
        order.id,
        exec_id="fractional-id-long",
        side="BUY",
        price=0.40,
        qty=1,
        ts=NOW,
        leg_con_id=575001,
    )
    transition(tmp_db, order.id, "filled", now=NOW)
    with tmp_db.begin() as conn:
        conn.execute(
            update(fills)
            .where(fills.c.ib_exec_id == "fractional-id-short")
            .values(leg_con_id=580001.5)
        )

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_cross_underlying_adoption(tmp_db: Engine) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 1001},
        {"symbol": "QQQ", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 95.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 1002},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="cross-short", side="SELL",
                price=1.40, qty=1, ts=NOW, leg_con_id=1001)
    record_fill(tmp_db, order_id, exec_id="cross-long", side="BUY",
                price=0.40, qty=1, ts=NOW, leg_con_id=1002)

    assert math.isinf(open_heat_dollars(tmp_db))


@pytest.mark.parametrize(
    ("currency", "multiplier"),
    [(None, 100), ("EUR", 100), ("USD", None), ("USD", 50), ("USD", float("nan"))],
)
def test_open_heat_fails_closed_without_supported_usd_contract_terms(
    tmp_db: Engine,
    currency: object,
    multiplier: object,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "P", "quantity": 1, "currency": currency,
         "multiplier": multiplier, "con_id": 2001},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 95.0, "right": "P", "quantity": 1, "currency": currency,
         "multiplier": multiplier, "con_id": 2002},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="terms-short", side="SELL",
                price=1.40, qty=1, ts=NOW, leg_con_id=2001)
    record_fill(tmp_db, order_id, exec_id="terms-long", side="BUY",
                price=0.40, qty=1, ts=NOW, leg_con_id=2002)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_misattributed_conid_fill(tmp_db: Engine) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 3001},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 95.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 3002},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="wrong-short", side="SELL",
                price=1.40, qty=1, ts=NOW, leg_con_id=9999)
    record_fill(tmp_db, order_id, exec_id="right-long", side="BUY",
                price=0.40, qty=1, ts=NOW, leg_con_id=3002)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_when_same_side_legs_lack_conid_attribution(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "TLT", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 85.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100},
        {"symbol": "TLT", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 88.0, "right": "C", "quantity": 1, "currency": "USD",
         "multiplier": 100},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="TLT", strategy="long_strangle",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="unknown-buy-a", side="BUY",
                price=0.60, qty=1, ts=NOW)
    record_fill(tmp_db, order_id, exec_id="unknown-buy-b", side="BUY",
                price=0.40, qty=1, ts=NOW)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_non_integral_order_quantity(tmp_db: Engine) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 4001},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 95.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 4002},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1.5, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="fraction-short", side="SELL",
                price=1.40, qty=1, ts=NOW, leg_con_id=4001)
    record_fill(tmp_db, order_id, exec_id="fraction-long", side="BUY",
                price=0.40, qty=1, ts=NOW, leg_con_id=4002)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_impossible_expiry_date(tmp_db: Engine) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260230",
         "strike": 100.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 5001},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260230",
         "strike": 95.0, "right": "P", "quantity": 1, "currency": "USD",
         "multiplier": 100, "con_id": 5002},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="date-short", side="SELL",
                price=1.40, qty=1, ts=NOW, leg_con_id=5001)
    record_fill(tmp_db, order_id, exec_id="date-long", side="BUY",
                price=0.40, qty=1, ts=NOW, leg_con_id=5002)

    assert math.isinf(open_heat_dollars(tmp_db))


@pytest.mark.parametrize("bad_max_loss", [float("nan"), -1.0])
def test_open_heat_fails_closed_for_invalid_scored_max_loss(
    tmp_db: Engine,
    bad_max_loss: float,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    score_id = _insert_pick(tmp_db, max_loss=bad_max_loss)
    with tmp_db.begin() as conn:
        conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            strategy_score_id=score_id, legs_json=CONDOR_LEGS,
            quantity=1, status="filled", staged_ts=NOW, submitted_ts=NOW,
            terminal_ts=NOW, reprice_count=0,
        ))

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_scoreless_undefined_risk(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="short_call",
            legs_json=[{
                "symbol": "SPY", "side": "sell", "sec_type": "OPT",
                "expiry": "20260731", "strike": 800.0, "right": "C", "quantity": 1,
            }],
            quantity=1, status="filled", staged_ts=NOW, submitted_ts=NOW,
            terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="adopt-short-call", side="SELL",
                price=1.00, qty=1, ts=NOW)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_when_adopted_fills_are_missing(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 95.0, "right": "P", "quantity": 1},
    ]
    with tmp_db.begin() as conn:
        conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1, limit_price=-1.00, status="filled",
            staged_ts=NOW, submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        ))

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_when_adopted_fills_are_incomplete(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 95.0, "right": "P", "quantity": 1},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="adopt-only-short", side="SELL",
                price=1.40, qty=1, ts=NOW)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_mixed_expiry_adoption(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    legs = [
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260731",
         "strike": 100.0, "right": "C", "quantity": 1},
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260831",
         "strike": 110.0, "right": "C", "quantity": 1},
    ]
    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="calendar_spread",
            legs_json=legs, quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="adopt-near-long", side="BUY",
                price=4.00, qty=1, ts=NOW)
    record_fill(tmp_db, order_id, exec_id="adopt-far-short", side="SELL",
                price=3.00, qty=1, ts=NOW)

    assert math.isinf(open_heat_dollars(tmp_db))


def test_open_heat_fails_closed_for_non_finite_adopted_leg(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.sizing import open_heat_dollars

    with tmp_db.begin() as conn:
        order_id = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="long_call",
            legs_json=[{
                "symbol": "SPY", "side": "buy", "sec_type": "OPT",
                "expiry": "20260731", "strike": float("nan"), "right": "C", "quantity": 1,
            }],
            quantity=1, status="filled", staged_ts=NOW, submitted_ts=NOW,
            terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key[0])
    record_fill(tmp_db, order_id, exec_id="adopt-nan-strike", side="BUY",
                price=1.00, qty=1, ts=NOW)

    assert math.isinf(open_heat_dollars(tmp_db))


async def test_engine_uses_dynamic_size_with_live_equity(tmp_db: Engine) -> None:
    # Pick: max_loss 380/unit, equity 100k → no-edge tilt ×0.5 → $1.5k → 3x.
    score_id = _insert_pick(tmp_db, suggested_quantity=9)
    deps = _deps(tmp_db, net_liquidation=100_000.0)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert outcome.ok, outcome.message
    call = deps.order_client.place_combo_limit.call_args
    assert call.kwargs["quantity"] == 3  # dynamic, NOT the indicative 9
    assert "sized 3x" in outcome.message


async def test_engine_persists_qualified_contract_terms_before_finalizing(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.orders import get_order
    from optionsbot.ibkr.types import PlacedOrder

    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    deps.order_client.place_combo_limit.side_effect = None
    deps.order_client.place_combo_limit.return_value = PlacedOrder(
        ib_order_id=11,
        order_ref="obot-qualified",
        action="BUY",
        limit_price=-1.20,
        quantity=1,
        leg_contracts=((580001, 100, "USD"), (575001, 100, "USD")),
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok, outcome.message
    persisted = get_order(tmp_db, outcome.order_id or 0)
    assert persisted is not None
    assert [leg["con_id"] for leg in persisted.legs] == [580001, 575001]


async def test_structural_max_loss_overrides_persisted_understatement(
    tmp_db: Engine,
) -> None:
    score_id = _insert_pick(tmp_db, max_loss=1.0)
    deps = _deps(
        tmp_db,
        net_liquidation=1_000.0,
        available_funds=1_000.0,
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert outcome.ok is False
    assert "max loss" in outcome.message.lower()
    deps.order_client.whatif_combo.assert_not_awaited()
    deps.order_client.place_combo_limit.assert_not_awaited()


async def test_engine_rejects_oversized_trade_for_small_account(tmp_db: Engine) -> None:
    # $18,885 max-loss wide put spread on a $5k account → clear refusal.
    wide_spread_legs = [
        {
            "symbol": "SPY",
            "side": "sell",
            "sec_type": "OPT",
            "expiry": "20260717",
            "strike": 580.0,
            "right": "P",
            "quantity": 1,
        },
        {
            "symbol": "SPY",
            "side": "buy",
            "sec_type": "OPT",
            "expiry": "20260717",
            "strike": 390.0,
            "right": "P",
            "quantity": 1,
        },
    ]
    score_id = _insert_pick(
        tmp_db,
        credit_or_debit=115.0,
        max_loss=18_885.0,
        max_profit=115.0,
        legs=wide_spread_legs,
    )
    deps = _deps(
        tmp_db,
        net_liquidation=5_000.0,
        available_funds=5_000.0,
        md_mids={(580.0, "P"): 1.60, (390.0, "P"): 0.45},
    )
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)
    assert not outcome.ok
    assert "single-trade cap" in outcome.message


async def test_engine_rejects_entry_when_realized_accounting_is_unavailable(
    tmp_db: Engine,
) -> None:
    entry_id, _ = _pair(tmp_db)
    with tmp_db.begin() as conn:
        conn.execute(delete(fills).where(fills.c.order_id == entry_id))
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=ENGINE_NOW)

    assert not outcome.ok
    assert "realized P&L accounting unavailable" in outcome.message
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


# --- IBK-131 realized pairs + report --------------------------------------------------


def _pair(
    engine: Engine, *, entry_credit: float = 1.20, close_debit: float = 0.50,
    commission: float = 0.65, closed_ts: datetime = NOW, strategy: str = "bull_put_spread",
) -> tuple[int, int]:
    entry_legs = [
        {
            "symbol": "SPY", "side": "sell", "sec_type": "OPT",
            "expiry": "20260731", "strike": 580.0, "right": "P", "quantity": 1,
            "con_id": 580001, "multiplier": 100, "currency": "USD",
        }
    ]
    close_legs = [{**entry_legs[0], "side": "buy"}]
    with engine.begin() as conn:
        epk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy=strategy, legs_json=entry_legs,
            quantity=1, status="filled", staged_ts=NOW - timedelta(days=5),
            submitted_ts=NOW - timedelta(days=5), terminal_ts=NOW - timedelta(days=5),
            reprice_count=0,
        )).inserted_primary_key
        assert epk is not None
        entry_id = int(epk[0])
        cpk = conn.execute(insert(orders).values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy=strategy, legs_json=close_legs, quantity=1, status="filled",
            staged_ts=closed_ts, submitted_ts=closed_ts, terminal_ts=closed_ts,
            reprice_count=0,
        )).inserted_primary_key
        assert cpk is not None
        close_id = int(cpk[0])
        for oid, ref in ((entry_id, f"obot-{entry_id}"), (close_id, f"obot-{close_id}")):
            conn.execute(update(orders).where(orders.c.id == oid).values(order_ref=ref))
    record_fill(engine, entry_id, exec_id=f"p{entry_id}", side="SELL",
                price=entry_credit, qty=1, ts=NOW - timedelta(days=5),
                leg_con_id=580001)
    record_fill(engine, close_id, exec_id=f"p{close_id}", side="BUY",
                price=close_debit, qty=1, ts=closed_ts, leg_con_id=580001)
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


def test_all_otm_expiration_settlement_is_durable_and_idempotent(
    tmp_db: Engine,
) -> None:
    legs = [
        {
            "symbol": "SPY",
            "side": "sell",
            "sec_type": "OPT",
            "expiry": "20260731",
            "strike": 580.0,
            "right": "P",
            "quantity": 1,
            "con_id": 580031,
            "multiplier": 100,
            "currency": "USD",
        }
    ]
    with tmp_db.begin() as conn:
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=legs,
                    quantity=1,
                    status="filled",
                    staged_ts=NOW,
                    submitted_ts=NOW,
                    terminal_ts=NOW,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
    record_fill(
        tmp_db,
        entry_id,
        exec_id="expiry-credit",
        side="SELL",
        price=1.20,
        qty=1,
        ts=NOW,
        leg_con_id=580031,
    )
    from optionsbot.execution.orders import set_fill_commission

    set_fill_commission(tmp_db, "expiry-credit", 0.65)
    entry = get_order(tmp_db, entry_id)
    assert entry is not None

    first = record_expired_worthless_settlement(
        tmp_db,
        entry,
        expiry="20260731",
        terminal_spot=600.0,
        settled_at=NOW,
    )
    second = record_expired_worthless_settlement(
        tmp_db,
        entry,
        expiry="20260731",
        terminal_spot=600.0,
        settled_at=NOW,
    )

    assert first == second
    assert first.close_id is None
    assert first.pnl == pytest.approx(119.35)
    assert settled_entry_ids(tmp_db) == {entry_id}
    assert realized_close_pairs(tmp_db) == [first]
    assert open_position_exposure(tmp_db) == {}


def test_expiration_settlement_rejects_any_in_the_money_leg(tmp_db: Engine) -> None:
    legs = [
        {
            "symbol": "SPY",
            "side": "sell",
            "sec_type": "OPT",
            "expiry": "20260731",
            "strike": 580.0,
            "right": "P",
            "quantity": 1,
            "con_id": 580032,
            "multiplier": 100,
            "currency": "USD",
        }
    ]
    with tmp_db.begin() as conn:
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=legs,
                    quantity=1,
                    status="filled",
                    staged_ts=NOW,
                    submitted_ts=NOW,
                    terminal_ts=NOW,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
    record_fill(
        tmp_db,
        entry_id,
        exec_id="expiry-itm",
        side="SELL",
        price=1.20,
        qty=1,
        ts=NOW,
        leg_con_id=580032,
    )
    from optionsbot.execution.orders import set_fill_commission

    set_fill_commission(tmp_db, "expiry-itm", 0.65)
    entry = get_order(tmp_db, entry_id)
    assert entry is not None

    with pytest.raises(RealizedPnLUnavailable, match="not provably all OTM"):
        record_expired_worthless_settlement(
            tmp_db,
            entry,
            expiry="20260731",
            terminal_spot=575.0,
            settled_at=NOW,
        )


def test_intrinsic_expiration_settlement_values_assignment(tmp_db: Engine) -> None:
    legs = [
        {
            "symbol": "SPY",
            "side": "sell",
            "sec_type": "OPT",
            "expiry": "20260731",
            "strike": 580.0,
            "right": "P",
            "quantity": 1,
            "con_id": 580033,
            "multiplier": 100,
            "currency": "USD",
        }
    ]
    with tmp_db.begin() as conn:
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="short_put",
                    legs_json=legs,
                    quantity=1,
                    status="filled",
                    staged_ts=NOW,
                    submitted_ts=NOW,
                    terminal_ts=NOW,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
    record_fill(
        tmp_db,
        entry_id,
        exec_id="expiry-itm-valued",
        side="SELL",
        price=1.20,
        qty=1,
        ts=NOW,
        leg_con_id=580033,
    )
    from optionsbot.execution.orders import set_fill_commission

    set_fill_commission(tmp_db, "expiry-itm-valued", 0.65)
    entry = get_order(tmp_db, entry_id)
    assert entry is not None

    pair = record_expiration_settlement(
        tmp_db,
        entry,
        expiry="20260731",
        terminal_spot=575.0,
        settled_at=NOW,
    )

    assert pair.pnl == pytest.approx(-380.65)
    assert expiration_assignment_shares(
        entry,
        expiry="20260731",
        terminal_spot=575.0,
    ) == 100
    with tmp_db.connect() as conn:
        settlement = conn.execute(
            select(position_settlements).where(
                position_settlements.c.entry_order_id == entry_id
            )
        ).one()
    assert settlement.kind == "expired_intrinsic"
    assert settlement.terminal_spot == 575.0
    assert open_position_exposure(tmp_db) == {}


def test_reconciliation_does_not_expect_expired_option_contracts(
    tmp_db: Engine,
) -> None:
    legs = [
        {
            "symbol": "SPY",
            "side": "sell",
            "sec_type": "OPT",
            "expiry": "20200102",
            "strike": 580.0,
            "right": "P",
            "quantity": 1,
            "con_id": 580020,
            "multiplier": 100,
            "currency": "USD",
        }
    ]
    with tmp_db.begin() as conn:
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=legs,
                    quantity=1,
                    status="filled",
                    staged_ts=NOW,
                    submitted_ts=NOW,
                    terminal_ts=NOW,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
    record_fill(
        tmp_db,
        entry_id,
        exec_id="cleared-expiry",
        side="SELL",
        price=1.20,
        qty=1,
        ts=NOW,
        leg_con_id=580020,
    )

    assert open_position_exposure(tmp_db) == {}


def test_realized_close_pairs_requires_exact_inverse_structure(tmp_db: Engine) -> None:
    _, close_id = _pair(tmp_db)
    with tmp_db.begin() as conn:
        close_legs = conn.execute(
            select(orders.c.legs_json).where(orders.c.id == close_id)
        ).scalar_one()
        malformed = [{**close_legs[0], "strike": 575.0}]
        conn.execute(
            update(orders).where(orders.c.id == close_id).values(legs_json=malformed)
        )

    with pytest.raises(RealizedPnLUnavailable, match="exact inverse"):
        realized_close_pairs(tmp_db)


def test_realized_close_pairs_rejects_fill_attributed_to_unrelated_contract(
    tmp_db: Engine,
) -> None:
    entry_id, _ = _pair(tmp_db)
    with tmp_db.begin() as conn:
        conn.execute(
            update(fills)
            .where(fills.c.order_id == entry_id)
            .values(leg_con_id=999999)
        )

    with pytest.raises(RealizedPnLUnavailable, match="contract attribution"):
        realized_close_pairs(tmp_db)


def test_realized_close_pairs_since_filter(tmp_db: Engine) -> None:
    _pair(tmp_db, closed_ts=NOW - timedelta(days=2))
    _pair(tmp_db, closed_ts=NOW)
    assert len(realized_close_pairs(tmp_db)) == 2
    assert len(realized_close_pairs(tmp_db, since=NOW - timedelta(hours=1))) == 1


@pytest.mark.parametrize(
    ("target", "mutation"),
    [
        ("entry", "missing_fill"),
        ("entry", "partial_fill"),
        ("close", "missing_fill"),
        ("close", "partial_fill"),
        ("entry", "missing_commission"),
        ("close", "missing_commission"),
    ],
)
def test_realized_close_pairs_is_unavailable_for_incomplete_accounting(
    tmp_db: Engine,
    target: str,
    mutation: str,
) -> None:
    entry_id, close_id = _pair(tmp_db)
    order_id = entry_id if target == "entry" else close_id
    with tmp_db.begin() as conn:
        if mutation == "missing_fill":
            conn.execute(delete(fills).where(fills.c.order_id == order_id))
        elif mutation == "partial_fill":
            conn.execute(update(orders).where(orders.c.id == order_id).values(quantity=2))
        else:
            conn.execute(
                update(fills).where(fills.c.order_id == order_id).values(commission=None)
            )

    with pytest.raises(RealizedPnLUnavailable, match=f"order {order_id}"):
        realized_close_pairs(tmp_db)


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


def test_loss_streak_suppresses_min1_floor() -> None:
    # IBK-Phase0: with the drawdown governor active (>=3 straight losses),
    # a trade whose sized quantity rounds to 0 must STAY 0 (skip), not get
    # floored back up to the min-1 lot.
    from optionsbot.execution.sizing import dynamic_quantity

    # $5k equity, base 3% = $150; governor ×0.5 → $75 budget; $90 max loss
    # → floor(75/90) = 0. Pre-fix this floored to 1; now it must be 0.
    d = dynamic_quantity(
        equity=5_000, max_loss_unit=90, max_profit_unit=30, prob_profit=0.70,
        open_heat=0, recent_pnls=[-10, -20, -5], base_risk_pct=0.03,
        heat_cap_pct=0.15, single_trade_cap_pct=0.10,
    )
    assert d.quantity == 0
    assert "loss streak" in d.note


def test_no_loss_streak_still_floors_to_min1() -> None:
    # Same fitting trade with NO loss streak must still floor to the
    # minimum-viable 1 lot (it fits both caps).
    from optionsbot.execution.sizing import dynamic_quantity

    d = dynamic_quantity(
        equity=5_000, max_loss_unit=90, max_profit_unit=30, prob_profit=0.70,
        open_heat=0, recent_pnls=[], base_risk_pct=0.03, heat_cap_pct=0.15,
        single_trade_cap_pct=0.10,
    )
    assert d.quantity == 1 and "min-1" in d.note
