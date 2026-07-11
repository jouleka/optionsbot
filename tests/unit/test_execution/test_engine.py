"""Tests for the /execute orchestration engine (IBK-126)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Engine, insert, select

from optionsbot.config import Settings
from optionsbot.execution.engine import ExecutionDeps, combo_mid, execute_pick
from optionsbot.execution.orders import get_order, stage_order, transition
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.ibkr.types import AccountSummary, MarginPreview, OptionQuote, PlacedOrder
from optionsbot.storage.schema import execution_state, snapshots, strategy_scores

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)

CONDOR_LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
]

QUOTE_MIDS = {(580.0, "P"): 1.60, (575.0, "P"): 0.40}  # fresh net credit 1.20


def _quote(
    strike: float,
    right: str,
    mid: float | None,
    *,
    delayed: bool = False,
    ts: datetime | None = NOW,
) -> OptionQuote:
    bid = round(mid - 0.05, 4) if mid is not None else None
    ask = round(mid + 0.05, 4) if mid is not None else None
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right=right,  # type: ignore[arg-type]
        bid=bid, ask=ask, last=None, mid=mid, iv=None, delta=None, gamma=None,
        theta=None, vega=None, open_interest=None, volume=None, ts=ts, delayed=delayed,
    )


def _insert_pick(
    engine: Engine,
    *,
    ts: datetime = NOW,
    symbol: str = "SPY",
    defined_risk: bool = True,
    suggested_quantity: int = 1,
    credit_or_debit: float = 120.0,  # dollars per set; 1.20/unit
    max_loss: float = 380.0,
    legs: list[dict[str, Any]] | None = None,
    raw_json: Any = None,
) -> int:
    with engine.begin() as conn:
        snapshot_id = conn.execute(
            insert(snapshots).values(
                symbol=symbol, ts=ts, spot=600.0, raw_json=raw_json
            )
        ).inserted_primary_key[0]
        score_id = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snapshot_id, strategy="bull_put_spread", score=78.0,
                rationale="t", legs_json=legs if legs is not None else CONDOR_LEGS,
                suggestion_json={
                    "defined_risk": defined_risk,
                    "credit_or_debit": credit_or_debit,
                    "max_loss": max_loss, "max_profit": 120.0, "prob_profit": 0.7,
                    "suggested_quantity": suggested_quantity,
                    "reward_risk": 0.32, "expected_value": 11.0,
                    "risk_tier": "balanced",
                },
            )
        ).inserted_primary_key[0]
    return int(score_id)


def _deps(
    tmp_db: Engine,
    *,
    enabled: bool = True,
    md_mids: dict[tuple[float, str], float | None] | None = None,
    available_funds: float | None = 50_000.0,
    net_liquidation: float | None = 50_000.0,
    account_currency: str = "USD",
    fx_to_usd: float = 1.0,
    margin_change: float | None = 380.0,
    walk: bool = False,
    delayed: bool = False,
    quote_ts: datetime | None = NOW,
) -> ExecutionDeps:
    settings = Settings()
    settings.execution.enabled = enabled
    with tmp_db.connect() as conn:
        day_start = conn.execute(
            select(execution_state.c.day_start_net_liq).where(execution_state.c.id == 1)
        ).scalar_one_or_none()
    if net_liquidation is not None and day_start is None:
        from optionsbot.execution.equity_guard import capture_day_start_net_liq

        capture_day_start_net_liq(tmp_db, float(net_liquidation))

    order_client = MagicMock()
    order_client.place_combo_limit = AsyncMock(
        side_effect=lambda *a, **k: PlacedOrder(
            ib_order_id=11, order_ref=k["order_ref"], action="BUY",
            limit_price=k["limit_price"], quantity=k["quantity"],
        )
    )
    order_client.whatif_combo = AsyncMock(
        return_value=MarginPreview(
            init_margin_change=margin_change, maint_margin_change=margin_change,
            equity_with_loan_change=None, commission=1.3, max_commission=None,
            warning=None,
        )
    )

    mids = md_mids if md_mids is not None else QUOTE_MIDS
    md = MagicMock()
    md.get_option_snapshot = AsyncMock(
        side_effect=lambda symbol, expiry, strike, right: _quote(
            strike, right, mids.get((strike, right)), delayed=delayed, ts=quote_ts
        )
    )

    order_client.modify_price = AsyncMock()
    order_client.cancel = AsyncMock()

    positions = MagicMock()
    positions.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=Decimal(str(net_liquidation)) if net_liquidation is not None else None,
            buying_power=None,
            available_funds=Decimal(str(available_funds)) if available_funds is not None else None,
            currency=account_currency,
            fx_to_usd=Decimal(str(fx_to_usd)),
        )
    )
    return ExecutionDeps(
        engine=tmp_db, settings=settings, order_client=order_client,
        md=md, positions=positions, ibkr_lock=asyncio.Lock(),
        walk_md=md if walk else None,
        walk_tasks=set() if walk else None,
    )


# --- combo_mid ------------------------------------------------------------------


def test_combo_mid_signed_credit_positive() -> None:
    quotes = {
        ("20260717", 580.0, "P"): _quote(580.0, "P", 1.60),
        ("20260717", 575.0, "P"): _quote(575.0, "P", 0.40),
    }
    assert combo_mid(CONDOR_LEGS, quotes) == pytest.approx(1.20)


def test_combo_mid_missing_quote_returns_none() -> None:
    quotes = {("20260717", 580.0, "P"): _quote(580.0, "P", 1.60)}
    assert combo_mid(CONDOR_LEGS, quotes) is None


def test_combo_mid_skips_stk_legs() -> None:
    legs = [
        {"symbol": "SPY", "side": "buy", "sec_type": "STK", "quantity": 100},
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 580.0, "right": "P", "quantity": 1},
    ]
    quotes = {("20260717", 580.0, "P"): _quote(580.0, "P", 1.60)}
    assert combo_mid(legs, quotes) == pytest.approx(1.60)


# --- gate rejections ---------------------------------------------------------------


async def test_rejects_when_not_armed(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    outcome = await execute_pick(_deps(tmp_db, enabled=False), score_id, now=NOW)
    assert not outcome.ok
    assert "enabled" in outcome.message


async def test_rejects_when_killed(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    trip_kill(tmp_db, "drawdown")
    outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "kill" in outcome.message.lower()


async def test_rejects_unknown_pick(tmp_db: Engine) -> None:
    outcome = await execute_pick(_deps(tmp_db), 999_999, now=NOW)
    assert not outcome.ok
    assert "unknown pick" in outcome.message.lower()


async def test_rejects_stale_pick(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, ts=NOW - timedelta(minutes=45))
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "stale" in outcome.message.lower()


@pytest.mark.parametrize(
    "raw_json",
    [
        None,
        {},
        {"delayed": False},
        {"warming_up": False},
        {"delayed": 0, "warming_up": False},
        {"delayed": False, "warming_up": 0},
        {"delayed": "false", "warming_up": False},
        {"delayed": False, "warming_up": "false"},
        [],
        "legacy",
    ],
)
async def test_auto_entry_rejects_snapshot_without_exact_ready_flags(
    tmp_db: Engine, raw_json: Any
) -> None:
    score_id = _insert_pick(tmp_db, raw_json=raw_json)
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "snapshot" in outcome.message.lower() or "data" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_auto_entry_accepts_exact_false_ready_flags(tmp_db: Engine) -> None:
    score_id = _insert_pick(
        tmp_db,
        raw_json={"delayed": False, "warming_up": False},
    )
    deps = _deps(tmp_db)
    deps.settings.execution.mode = "auto"

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert outcome.ok, outcome.message
    deps.order_client.place_combo_limit.assert_awaited_once()  # type: ignore[attr-defined]


async def test_rejects_undefined_risk(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, defined_risk=False)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "undefined risk" in outcome.message.lower()


async def test_rejects_non_finite_max_loss(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, max_loss=float("nan"))
    deps = _deps(tmp_db)

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "defined max loss" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_rejects_option_quote_with_unknown_timestamp(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, quote_ts=None)

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "timestamp" in outcome.message.lower() or "quote age" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_rejects_stale_option_quote_timestamp(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, quote_ts=NOW - timedelta(seconds=46))

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "quote age" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_rejects_delayed_option_quotes(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, delayed=True)

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "delayed" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_rejects_zero_quantity(tmp_db: Engine) -> None:
    # With dynamic sizing, a pick whose max_loss exceeds the single-trade cap
    # on a tiny account should be rejected (not placed). $500 equity, $380
    # max_loss: single-trade cap = $500 * 10% = $50 < $380 → rejects.
    score_id = _insert_pick(tmp_db, suggested_quantity=1)
    deps = _deps(tmp_db, net_liquidation=500.0, available_funds=500.0)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "cap" in outcome.message.lower() or "sized" in outcome.message.lower()


async def test_entry_drawdown_uses_usd_net_liquidation_for_non_usd_account(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    # Baseline is persisted in USD. EUR 8k at 1.25 USD/EUR is unchanged
    # $10k equity; comparing the raw EUR value would invent a 20% drawdown.
    capture_day_start_net_liq(tmp_db, 10_000.0)
    score_id = _insert_pick(tmp_db)
    deps = _deps(
        tmp_db,
        net_liquidation=8_000.0,
        available_funds=8_000.0,
        account_currency="EUR",
        fx_to_usd=1.25,
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert outcome.ok, outcome.message
    deps.order_client.place_combo_limit.assert_awaited_once()


async def test_rejects_when_live_equity_unavailable(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, suggested_quantity=9)
    deps = _deps(tmp_db, net_liquidation=None)  # net-liq read failed
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "equity" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()


async def test_rejects_when_market_closed(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=False):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "market" in outcome.message.lower()


async def test_rejects_duplicate_active_order(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    stage_order(tmp_db, score_id, now=NOW)  # active order exists for this pick
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "already" in outcome.message.lower()


async def test_rejects_reexecute_after_any_terminal_order_intent(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    record = stage_order(tmp_db, score_id, now=NOW)
    transition(tmp_db, record.id, "skipped", now=NOW)
    deps = _deps(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "authorization is consumed" in outcome.message


async def test_rejects_at_max_open_positions(tmp_db: Engine) -> None:
    deps = _deps(tmp_db)
    deps.settings.execution.max_open_positions = 1
    other = _insert_pick(tmp_db, symbol="QQQ")
    record = stage_order(tmp_db, other, now=NOW)
    transition(tmp_db, record.id, "submitting", now=NOW)
    score_id = _insert_pick(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "position" in outcome.message.lower()


async def test_rejects_at_max_per_symbol(tmp_db: Engine) -> None:
    deps = _deps(tmp_db)
    deps.settings.execution.max_open_positions = 10
    first = _insert_pick(tmp_db)
    stage_order(tmp_db, first, now=NOW)  # SPY order active
    second = _insert_pick(tmp_db)  # another SPY pick
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, second, now=NOW)
    assert not outcome.ok
    assert "SPY" in outcome.message


async def test_rejects_when_leg_quote_missing(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, md_mids={(580.0, "P"): 1.60, (575.0, "P"): None})
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "quote" in outcome.message.lower()


async def test_rejects_when_credit_sign_flipped(tmp_db: Engine) -> None:
    # Scan said credit; fresh quotes now net a DEBIT — the edge is gone.
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, md_mids={(580.0, "P"): 0.30, (575.0, "P"): 0.40})
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "edge" in outcome.message.lower() or "credit" in outcome.message.lower()


async def test_rejects_when_margin_exceeds_available(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, available_funds=100.0, margin_change=380.0)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "margin" in outcome.message.lower()


async def test_rejects_when_whatif_margin_unknown(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, margin_change=None)  # IBKR returned no init-margin
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "margin" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()


async def test_rejects_when_available_funds_unknown(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, available_funds=None)  # type: ignore[arg-type]
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "margin" in outcome.message.lower() or "funds" in outcome.message.lower()


@pytest.mark.parametrize("margin_change", [float("nan"), float("inf"), float("-inf")])
async def test_rejects_non_finite_final_margin_requirement(
    tmp_db: Engine, margin_change: float
) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, margin_change=margin_change)

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "margin" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.parametrize("available_funds", [float("nan"), float("inf"), float("-inf")])
async def test_rejects_non_finite_final_available_funds(
    tmp_db: Engine, available_funds: float
) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, available_funds=available_funds)

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert not outcome.ok
    assert "margin" in outcome.message.lower() or "funds" in outcome.message.lower()
    deps.order_client.place_combo_limit.assert_not_awaited()  # type: ignore[attr-defined]


async def test_rejects_when_whatif_raises(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    deps.order_client.whatif_combo = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "whatif" in outcome.message.lower() or "margin" in outcome.message.lower()


# --- happy path + place failure ------------------------------------------------------


async def test_happy_path_stages_places_and_reports(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok, outcome.message
    assert outcome.order_id is not None

    record = get_order(tmp_db, outcome.order_id)
    assert record is not None
    assert record.status == "submitted"
    assert record.ib_order_id == 11
    assert record.strategy_score_id == score_id

    call = deps.order_client.place_combo_limit.call_args
    assert call.kwargs["order_ref"] == f"obot-{record.id}"
    assert call.kwargs["quantity"] == 1
    # fresh net credit 1.20/unit -> BUY-bag limit is NEGATIVE 1.20
    assert call.kwargs["limit_price"] == pytest.approx(-1.20)
    assert f"#{record.id}" in outcome.message
    assert "1.20" in outcome.message


async def test_limit_price_rounds_to_tick(tmp_db: Engine) -> None:
    # Half-cent mids (bid 9.30/ask 9.33 -> 9.315) must round to the penny —
    # IBKR rejects sub-increment limits with Error 110 (seen live, pick 686).
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, md_mids={(580.0, "P"): 1.615, (575.0, "P"): 0.40})
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok, outcome.message
    call = deps.order_client.place_combo_limit.call_args
    limit = call.kwargs["limit_price"]
    assert limit == pytest.approx(round(limit, 2))  # on the penny grid
    # Which side of the half-tick it lands on is float noise; within half a
    # tick of the true mid is the contract.
    assert abs(-limit - 1.215) <= 0.005 + 1e-9


async def test_happy_path_debit_places_positive_limit(tmp_db: Engine) -> None:
    # Debit spread: scan says we PAY (credit_or_debit < 0); fresh quotes net a
    # debit too. BUY-bag debit = POSITIVE limit — the most error-prone sign in
    # the codebase, pinned here.
    score_id = _insert_pick(tmp_db, credit_or_debit=-120.0)
    deps = _deps(tmp_db, md_mids={(580.0, "P"): 0.40, (575.0, "P"): 1.60})
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok, outcome.message
    call = deps.order_client.place_combo_limit.call_args
    assert call.kwargs["limit_price"] == pytest.approx(+1.20)
    assert "debit" in outcome.message


async def test_drift_warning_included(tmp_db: Engine) -> None:
    # Scan credit $1.80/unit, fresh mid $1.20 -> 33% drift > 25% default band.
    score_id = _insert_pick(tmp_db, credit_or_debit=180.0)
    deps = _deps(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok
    assert "drift" in outcome.message.lower()


async def test_rejects_illiquid_wide_spread(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    # Stub legs quote a $0.10 spread; force a per-leg failure with a tiny cap.
    deps.settings.execution.max_leg_spread_frac = 0.01
    deps.settings.execution.max_leg_spread_floor = 0.01
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "liquidity" in outcome.message.lower()


async def test_rejects_when_combo_spread_eats_the_credit(tmp_db: Engine) -> None:
    # Per-leg spreads pass, but the combo bid/ask is too big a fraction of the
    # net premium -> economic gate rejects (slippage would eat the edge).
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    deps.settings.execution.max_combo_spread_frac = 0.01  # combo spread > 1% of net
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "liquidity" in outcome.message.lower() and "net premium" in outcome.message


async def test_decision_quotes_journaled(tmp_db: Engine) -> None:
    from sqlalchemy import select

    from optionsbot.storage.schema import order_quotes

    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok
    with tmp_db.connect() as conn:
        row = conn.execute(select(order_quotes)).one()
    assert row.kind == "decision"
    assert row.order_id == outcome.order_id
    assert row.combo_mid == pytest.approx(1.20)
    assert row.legs_json and row.legs_json[0]["bid"] is not None


async def test_happy_path_spawns_walk_task(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, walk=True)
    deps.settings.execution.walk_step_seconds = 0
    deps.settings.execution.walk_max_steps = 2
    deps.settings.execution.walk_final_rest_seconds = 0
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok
    assert deps.walk_tasks

    from sqlalchemy import update as sa_update

    from optionsbot.storage.schema import orders as orders_table

    async def confirm(ib_order_id: int) -> None:  # the tracker's job
        with tmp_db.begin() as conn:
            conn.execute(
                sa_update(orders_table)
                .where(orders_table.c.id == outcome.order_id)
                .values(status="cancelled", terminal_ts=NOW)
            )

    deps.order_client.cancel = AsyncMock(side_effect=confirm)
    await asyncio.gather(*list(deps.walk_tasks))
    # The stubbed order never fills, so the walk repriced then requested
    # the cancel; the simulated tracker confirmed it.
    assert deps.order_client.modify_price.await_count == 2
    deps.order_client.cancel.assert_awaited_once()
    record = get_order(tmp_db, outcome.order_id)  # type: ignore[arg-type]
    assert record is not None
    assert record.status == "cancelled"
    assert "walk exhausted" in (record.last_error or "")


async def test_no_walk_spawned_without_walk_md(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)  # walk_md None — v1 place-at-mid behavior
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok
    deps.order_client.modify_price.assert_not_awaited()


async def test_reconcile_race_cancels_at_broker(tmp_db: Engine) -> None:
    # Opus C1 backstop: if a reconcile pass resolves the row terminal while
    # place is in flight, the just-placed REAL order must be pulled.
    from sqlalchemy import update as sa_update

    from optionsbot.storage.schema import orders as orders_table

    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)

    async def race_place(*args: object, **kwargs: object) -> PlacedOrder:
        # Simulate reconcile flipping the row to skipped mid-place.
        with tmp_db.begin() as conn:
            conn.execute(
                sa_update(orders_table)
                .where(orders_table.c.status == "submitting")
                .values(status="skipped", terminal_ts=NOW)
            )
        return PlacedOrder(
            ib_order_id=77, order_ref=str(kwargs["order_ref"]), action="BUY",
            limit_price=float(kwargs["limit_price"]), quantity=int(kwargs["quantity"]),
        )

    deps.order_client.place_combo_limit = AsyncMock(side_effect=race_place)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "race" in outcome.message.lower()
    deps.order_client.cancel.assert_awaited_once_with(77)
    state = load_state(tmp_db)
    assert state.killed is True
    assert state.reason is not None and "broker" in state.reason


async def test_place_failure_halts_and_preserves_submitting_claim(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    deps.order_client.place_combo_limit = AsyncMock(
        side_effect=RuntimeError("gateway acknowledgement lost")
    )
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "outcome unknown" in outcome.message
    assert outcome.order_id is not None
    record = get_order(tmp_db, outcome.order_id)
    assert record is not None
    assert record.status == "submitting"
    assert record.last_error is not None and "unknown" in record.last_error
    state = load_state(tmp_db)
    assert state.killed is True
    assert state.reason is not None and "unknown" in state.reason


async def test_execute_pick_rechecks_drawdown_on_final_sizing_summary(
    tmp_db: Engine,
) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db, net_liquidation=100_000.0)
    deps.positions.get_account_summary = AsyncMock(
        side_effect=[
            AccountSummary(
                net_liquidation=Decimal("100000"),
                buying_power=None,
                available_funds=Decimal("50000"),
                currency="USD",
                fx_to_usd=Decimal("1"),
            ),
            AccountSummary(
                net_liquidation=Decimal("90000"),
                buying_power=None,
                available_funds=Decimal("50000"),
                currency="USD",
                fx_to_usd=Decimal("1"),
            ),
        ]
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert outcome.ok is False
    assert "drawdown" in outcome.message.lower()
    deps.order_client.whatif_combo.assert_not_awaited()
    deps.order_client.place_combo_limit.assert_not_awaited()


async def test_execute_pick_blocks_near_daily_loss_cap(tmp_db: Engine) -> None:
    # PHASE 0 B1: once the day-start drawdown reaches entry_block_loss_frac of the
    # cap, execute_pick rejects BEFORE staging — even with a valid fresh pick.
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    score_id = _insert_pick(tmp_db)
    # Pre-capture day-start baseline of 100k. 98.4k = 1.6% down; block at 75% of 2% = 1.5%.
    capture_day_start_net_liq(tmp_db, 100_000.0, session="2026-06-24")
    deps = _deps(tmp_db, net_liquidation=98_400.0)
    deps.settings.execution.entry_block_loss_frac = 0.75
    deps.settings.execution.max_daily_loss_pct = 0.02
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok is False
    assert "drawdown" in outcome.message.lower()


async def test_entry_drawdown_compares_usd_values_for_non_usd_account(
    tmp_db: Engine,
) -> None:
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    capture_day_start_net_liq(tmp_db, 10_000.0, session="2026-06-24")
    score_id = _insert_pick(tmp_db)
    deps = _deps(
        tmp_db,
        net_liquidation=8_000.0,
        available_funds=8_000.0,
        account_currency="EUR",
        fx_to_usd=1.25,
    )

    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)

    assert outcome.ok, outcome.message
    deps.order_client.place_combo_limit.assert_awaited_once()  # type: ignore[attr-defined]
