"""Tests for the /execute orchestration engine (IBK-126)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Engine, insert

from optionsbot.config import Settings
from optionsbot.execution.engine import ExecutionDeps, combo_mid, execute_pick
from optionsbot.execution.orders import get_order, stage_order, transition
from optionsbot.execution.state import trip_kill
from optionsbot.ibkr.types import AccountSummary, MarginPreview, OptionQuote, PlacedOrder
from optionsbot.storage.schema import snapshots, strategy_scores

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)

CONDOR_LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
]

QUOTE_MIDS = {(580.0, "P"): 1.60, (575.0, "P"): 0.40}  # fresh net credit 1.20


def _quote(strike: float, right: str, mid: float | None) -> OptionQuote:
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right=right,  # type: ignore[arg-type]
        bid=None, ask=None, last=None, mid=mid, iv=None, delta=None, gamma=None,
        theta=None, vega=None, open_interest=None, volume=None, ts=NOW, delayed=True,
    )


def _insert_pick(
    engine: Engine,
    *,
    ts: datetime = NOW,
    symbol: str = "SPY",
    defined_risk: bool = True,
    suggested_quantity: int = 1,
    credit_or_debit: float = 120.0,  # dollars per set; 1.20/unit
    legs: list[dict[str, Any]] | None = None,
) -> int:
    with engine.begin() as conn:
        snapshot_id = conn.execute(
            insert(snapshots).values(symbol=symbol, ts=ts, spot=600.0)
        ).inserted_primary_key[0]
        score_id = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snapshot_id, strategy="bull_put_spread", score=78.0,
                rationale="t", legs_json=legs if legs is not None else CONDOR_LEGS,
                suggestion_json={
                    "defined_risk": defined_risk,
                    "credit_or_debit": credit_or_debit,
                    "max_loss": 380.0, "max_profit": 120.0, "prob_profit": 0.7,
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
    available_funds: float = 50_000.0,
    margin_change: float | None = 380.0,
) -> ExecutionDeps:
    settings = Settings()
    settings.execution.enabled = enabled

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
            strike, right, mids.get((strike, right))
        )
    )

    positions = MagicMock()
    positions.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=None, buying_power=None,
            available_funds=available_funds, currency="USD",  # type: ignore[arg-type]
        )
    )
    return ExecutionDeps(
        engine=tmp_db, settings=settings, order_client=order_client,
        md=md, positions=positions, ibkr_lock=asyncio.Lock(),
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


async def test_rejects_undefined_risk(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, defined_risk=False)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "undefined risk" in outcome.message.lower()


async def test_rejects_zero_quantity(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db, suggested_quantity=0)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert not outcome.ok
    assert "quantity" in outcome.message.lower()


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


async def test_allows_reexecute_after_failed_terminal(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    record = stage_order(tmp_db, score_id, now=NOW)
    transition(tmp_db, record.id, "skipped", now=NOW)  # earlier attempt failed gates
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(_deps(tmp_db), score_id, now=NOW)
    assert outcome.ok


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


async def test_drift_warning_included(tmp_db: Engine) -> None:
    # Scan credit $1.80/unit, fresh mid $1.20 -> 33% drift > 25% default band.
    score_id = _insert_pick(tmp_db, credit_or_debit=180.0)
    deps = _deps(tmp_db)
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert outcome.ok
    assert "drift" in outcome.message.lower()


async def test_place_failure_marks_skipped(tmp_db: Engine) -> None:
    score_id = _insert_pick(tmp_db)
    deps = _deps(tmp_db)
    deps.order_client.place_combo_limit = AsyncMock(
        side_effect=RuntimeError("gateway exploded")
    )
    with patch("optionsbot.execution.engine.is_market_open", return_value=True):
        outcome = await execute_pick(deps, score_id, now=NOW)
    assert not outcome.ok
    assert "gateway exploded" in outcome.message
    assert outcome.order_id is not None
    record = get_order(tmp_db, outcome.order_id)
    assert record is not None
    assert record.status == "skipped"
    assert record.last_error == "gateway exploded"
