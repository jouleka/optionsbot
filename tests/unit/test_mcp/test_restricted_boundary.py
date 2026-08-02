"""Negative tests for the Hermes least-privilege MCP boundary (IBK-137)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.exc import OperationalError

from optionsbot.config import Settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.control_intents import (
    _consume_entry_proposal,
    _consume_entry_review,
    consume_control_intents,
    consume_control_intents_async,
)
from optionsbot.execution.state import load_state
from optionsbot.mcp_server.intent_queue import (
    control_intents,
    create_intent_engine,
    enqueue_intent,
    recent_proposal_decisions,
)
from optionsbot.mcp_server.restricted_context import RestrictedServerContext
from optionsbot.mcp_server.server import build_server
from optionsbot.mcp_server.tools import nightwatch, restricted
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.storage.db import create_readonly_engine_for_path
from optionsbot.storage.schema import (
    alerts,
    entry_reviews,
    exit_requests,
    hermes_overlay_state,
    orders,
    pick_outcomes,
    snapshots,
    strategy_scores,
    watchlist,
)
from optionsbot.strategies import StrategySuggestion
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


@pytest.mark.asyncio
async def test_restricted_server_exposes_only_bounded_surface() -> None:
    server = build_server(restricted=True)
    names = {tool.name for tool in await server.list_tools()}
    assert names == {
        "analyze",
        "control_intent_status",
        "daily_brief",
        "halt",
        "health",
        "hermes_metrics",
        "latest_snapshot",
        "list_watchlist",
        "pick_review_packet",
        "pending_picks",
        "positions",
        "propose_entry",
        "request_exit",
        "score_breakdown",
        "submit_entry_review",
        "track_record",
    }
    assert {"add_to_watchlist", "remove_from_watchlist", "set_view_override"}.isdisjoint(names)


def test_restricted_health_attests_zero_dte_entry_window(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    context = RestrictedServerContext(
        engine=mcp_engine,
        intent_engine=create_intent_engine(tmp_path / "health-intents.db"),
        max_pick_age_minutes=20,
        zero_dte_only=True,
        zero_dte_entry_cutoff_minutes=90,
    )
    with (
        patch(
            "optionsbot.mcp_server.tools.restricted.is_market_open",
            return_value=True,
        ),
        patch(
            "optionsbot.mcp_server.tools.restricted.minutes_to_nyse_close",
            return_value=180.0,
        ),
    ):
        result = get_tools(restricted.register)["health"](FakeCtx(context))

    assert result["market_timing"] == {
        "nyse_session": result["market_timing"]["nyse_session"],
        "market_open": True,
        "minutes_to_close": 180.0,
        "zero_dte_only": True,
        "zero_dte_entry_cutoff_minutes": 90,
        "opening_range_fvg_enabled": False,
        "opening_range_minutes": 10,
        "opening_range_entry_window_minutes": 90,
        "opening_range_window_open": True,
        "entry_window_open": True,
    }


def test_hermes_metrics_uses_only_judgeable_calls_and_reports_churn(
    mcp_engine: Engine,
) -> None:
    with mcp_engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=datetime.now(UTC),
                    spot=500.0,
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        score_ids = [
            int(
                conn.execute(
                    insert(strategy_scores).values(
                        snapshot_id=snapshot_id,
                        strategy=strategy,
                        score=80.0,
                        legs_json=[],
                        suggestion_json={"review_evidence": {"ready": True}},
                    )
                ).inserted_primary_key[0]
            )
            for strategy in ("bull_call_spread", "bear_put_spread", "iron_condor")
        ]
        alert_ids = [
            int(
                conn.execute(
                    insert(alerts).values(
                        strategy_score_id=score_id,
                        ts=datetime.now(UTC),
                        symbol="SPY",
                        strategy=strategy,
                        score=80.0,
                        status="sent",
                        sent_ts=datetime.now(UTC),
                        telegram_msg_id=100 + index,
                    )
                ).inserted_primary_key[0]
            )
            for index, (score_id, strategy) in enumerate(
                zip(
                    score_ids,
                    ("bull_call_spread", "bear_put_spread", "iron_condor"),
                    strict=True,
                )
            )
        ]
        for score_id, alert_id, verdict, status in (
            (score_ids[0], alert_ids[0], "vetted_paper_candidate", "submitted"),
            (score_ids[1], alert_ids[1], "no_trade", "refused"),
            (score_ids[2], alert_ids[2], "watch_only", "held"),
        ):
            conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=score_id,
                    alert_id=alert_id,
                    reviewed_at=datetime.now(UTC),
                    verdict=verdict,
                    confidence=0.8,
                    sources_json=["source A", "source B"],
                    reason="Persisted test review with enough audit detail.",
                    checks_json={"candidate": True},
                    status=status,
                )
            )
        for score_id, strategy, pnl, win in (
            (score_ids[0], "bull_call_spread", 125.0, 1),
            (score_ids[1], "bear_put_spread", 40.0, 1),
            (score_ids[2], "iron_condor", 20.0, 1),
        ):
            conn.execute(
                insert(pick_outcomes).values(
                    strategy_score_id=score_id,
                    symbol="SPY",
                    strategy=strategy,
                    expiry="2026-07-17",
                    entry_spot=500.0,
                    terminal_spot=505.0,
                    realized_pnl=pnl,
                    win=win,
                    evaluated_at=datetime.now(UTC),
                )
            )
        position_id = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="bull_call_spread",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=datetime.now(UTC),
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(exit_requests).values(
                position_id=position_id,
                requested_at=datetime.now(UTC),
                catalyst_type="news",
                confidence=0.8,
                sources_json=["source A", "source B"],
                reason="Test exit request",
                status="refused",
            )
        )

    metrics = get_tools(restricted.register)["hermes_metrics"](
        FakeCtx(SimpleNamespace(engine=mcp_engine))
    )

    correctness = metrics["entry_overlay_correctness"]
    assert correctness["calls"] == 3
    assert correctness["judgeable"] == 3
    assert correctness["useful"] == 1
    assert correctness["accuracy"] == 1 / 3
    assert correctness["threshold"] == 0.5
    assert correctness["payoff_efficiency_threshold"] == 0.5
    assert correctness["recommendation"] == "KEEP"
    assert correctness["small_sample"] is True
    assert correctness["unmatched_calls"] == 0
    assert correctness["payoff"] == {
        "captured_profit": 125.0,
        "avoided_loss": 0.0,
        "missed_profit": 60.0,
        "incurred_loss": 0.0,
        "helpful_dollars": 125.0,
        "harmful_dollars": 60.0,
        "decision_value": 65.0,
        "payoff_efficiency": 125 / 185,
    }
    assert correctness["by_verdict"]["no_trade"]["accuracy"] == 0.0
    assert correctness["by_verdict"]["vetted_paper_candidate"]["accuracy"] == 1.0
    assert correctness["by_verdict"]["watch_only"]["accuracy"] == 0.0
    assert metrics["request_churn"]["entry_reviews_by_status"] == {
        "held": 1,
        "refused": 1,
        "submitted": 1,
    }
    assert metrics["request_churn"]["exit_requests_by_status"] == {"refused": 1}


def test_hermes_metrics_zero_denominator_is_not_zero_percent(
    mcp_engine: Engine,
) -> None:
    metrics = get_tools(restricted.register)["hermes_metrics"](
        FakeCtx(SimpleNamespace(engine=mcp_engine))
    )

    correctness = metrics["entry_overlay_correctness"]
    assert correctness["judgeable"] == 0
    assert correctness["accuracy"] is None
    assert correctness["recommendation"] == "CHANGE"


def test_disabled_overlay_holds_newly_imported_vetted_review(
    mcp_engine: Engine,
) -> None:
    daemon_context = cast(
        DaemonContext,
        SimpleNamespace(engine=mcp_engine),
    )
    now = datetime.now(UTC)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=now, spot=600.0, raw_json={})
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        alert_id = int(
            conn.execute(
                insert(alerts).values(
                    strategy_score_id=score_id,
                    ts=now,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    score=80.0,
                    status="sent",
                    sent_ts=now,
                    telegram_msg_id=123,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            update(hermes_overlay_state)
            .where(hermes_overlay_state.c.id == 1)
            .values(
                enabled=0,
                reason="test correctness trip",
                ts=now,
                judgeable=20,
                accuracy=0.45,
            )
        )

    result = _consume_entry_review(
        daemon_context,
        {
            "pick_id": score_id,
            "alert_id": alert_id,
            "reviewed_at": now.isoformat(),
            "verdict": "vetted_paper_candidate",
            "confidence": 0.9,
            "sources": ["source A", "source B"],
            "reason": "test review",
            "checks": {
                "bot_health": True,
                "candidate": True,
                "microstructure": True,
                "greeks": True,
                "regime_history": True,
                "catalysts": True,
                "account_risk": True,
            },
        },
    )

    assert "held by the overlay breaker" in result
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"
    assert "test correctness trip" in review.decision_reason


def test_watch_only_result_does_not_claim_overlay_breaker(
    mcp_engine: Engine,
) -> None:
    daemon_context = cast(DaemonContext, SimpleNamespace(engine=mcp_engine))
    now = datetime.now(UTC)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=now, spot=600.0, raw_json={})
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        alert_id = int(
            conn.execute(
                insert(alerts).values(
                    strategy_score_id=score_id,
                    ts=now,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    score=80.0,
                    status="sent",
                    sent_ts=now,
                    telegram_msg_id=124,
                )
            ).inserted_primary_key[0]
        )

    result = _consume_entry_review(
        daemon_context,
        {
            "pick_id": score_id,
            "alert_id": alert_id,
            "reviewed_at": now.isoformat(),
            "verdict": "watch_only",
            "confidence": 0.9,
            "sources": ["source A", "source B"],
            "reason": "credible thesis with an explicit current guard",
            "checks": {"candidate": True, "microstructure": False},
        },
    )

    assert "watch-only; no order authority" in result
    assert "overlay breaker" not in result


def test_restricted_server_imports_no_broker_or_execution_modules() -> None:
    code = """
import sys
from optionsbot.mcp_server.server import build_server
build_server(restricted=True)
forbidden = sorted(
    name for name in sys.modules
    if name == 'optionsbot.ibkr' or name.startswith('optionsbot.ibkr.')
    or name == 'optionsbot.execution' or name.startswith('optionsbot.execution.')
)
if forbidden:
    raise SystemExit('forbidden imports: ' + ','.join(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_primary_database_engine_is_physically_read_only(
    mcp_engine: Engine, mcp_settings: Settings
) -> None:
    with mcp_engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))
    readonly = create_readonly_engine_for_path(mcp_settings.storage.db_path)
    try:
        with readonly.connect() as conn:
            assert conn.execute(select(watchlist.c.symbol)).scalar_one() == "SPY"
        with pytest.raises(OperationalError):
            with readonly.begin() as conn:
                conn.execute(insert(watchlist).values(symbol="QQQ", added_at=datetime.now(UTC)))
    finally:
        readonly.dispose()


def test_restricted_context_contains_no_broker_or_messaging_secrets(
    mcp_engine: Engine, tmp_path: Path
) -> None:
    intent_engine = create_intent_engine(tmp_path / "intents.db")
    context = RestrictedServerContext(
        engine=mcp_engine,
        intent_engine=intent_engine,
        max_pick_age_minutes=20,
    )
    assert context.broker_access is False
    assert not hasattr(context, "ibkr")
    assert not hasattr(context.settings, "telegram")
    assert not hasattr(context.settings, "ibkr")


def test_hermes_entry_proposal_only_enters_isolated_queue(
    mcp_engine: Engine, tmp_path: Path
) -> None:
    intent_engine = create_intent_engine(tmp_path / "proposal-intents.db")
    context = RestrictedServerContext(
        engine=mcp_engine,
        intent_engine=intent_engine,
        max_pick_age_minutes=20,
    )
    propose = get_tools(nightwatch.register)["propose_entry"]
    result = propose(
        "SPY",
        "bull",
        "neutral",
        "auto",
        0.9,
        ["trusted daemon", "Finnhub"],
        "Index strength supports a bounded bullish 0DTE structure.",
        {
            "bot_health": True,
            "regime_history": True,
            "catalysts": True,
        },
        FakeCtx(context),
    )

    assert result["ok"] is True
    assert result["status"] == "queued_for_optionsbot_validation"
    with intent_engine.connect() as conn:
        row = conn.execute(select(control_intents)).one()
    assert row.kind == "entry_proposal"
    assert row.status == "pending"
    assert row.payload_json["symbol"] == "SPY"
    assert row.payload_json["direction"] == "bull"
    with mcp_engine.connect() as conn:
        assert conn.execute(select(orders.c.id)).fetchall() == []


def test_terminal_proposal_decision_is_available_to_next_learning_pass(
    tmp_path: Path,
) -> None:
    intent_engine = create_intent_engine(tmp_path / "proposal-feedback.db")
    proposed_at = datetime.now(UTC)
    intent_id, _ = enqueue_intent(
        intent_engine,
        "entry_proposal",
        {
            "symbol": "QQQ",
            "direction": "bear",
            "iv_regime": "neutral",
            "strategy": "bear_put_spread",
            "confidence": 0.78,
        },
        now=proposed_at,
    )
    with intent_engine.begin() as conn:
        conn.execute(
            update(control_intents)
            .where(control_intents.c.id == intent_id)
            .values(
                status="processed",
                processed_at=proposed_at,
                result_text=(
                    "Hermes proposal declined as score #42: "
                    "non_positive_edge(expected_value=-12.50)"
                ),
            )
        )

    assert recent_proposal_decisions(intent_engine) == [
        {
            "intent_id": intent_id,
            "symbol": "QQQ",
            "direction": "bear",
            "iv_regime": "neutral",
            "strategy": "bear_put_spread",
            "confidence": 0.78,
            "proposed_at": proposed_at.isoformat(),
            "processed_at": proposed_at.isoformat(),
            "status": "processed",
            "decision": (
                "Hermes proposal declined as score #42: non_positive_edge(expected_value=-12.50)"
            ),
        }
    ]


async def test_async_consumer_routes_entry_proposal_to_live_rebuilder(
    mcp_engine: Engine, tmp_path: Path
) -> None:
    intent_path = tmp_path / "proposal-consumer.db"
    intent_engine = create_intent_engine(intent_path)
    enqueue_intent(
        intent_engine,
        "entry_proposal",
        {"symbol": "SPY"},
    )
    daemon_context = cast("DaemonContext", SimpleNamespace(engine=mcp_engine))
    with patch(
        "optionsbot.daemon.control_intents._consume_entry_proposal",
        new=AsyncMock(return_value="proposal independently rebuilt"),
    ) as rebuild:
        consumed = await consume_control_intents_async(daemon_context, intent_path)

    assert consumed == 1
    rebuild.assert_awaited_once()
    with intent_engine.connect() as conn:
        row = conn.execute(select(control_intents)).one()
    assert row.status == "processed"
    assert row.result_text == "proposal independently rebuilt"


@pytest.mark.asyncio
async def test_declined_entry_proposal_does_not_create_alertless_review(
    mcp_engine: Engine,
) -> None:
    now = datetime.now(UTC)
    with mcp_engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=now,
                    spot=500.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snapshot_id,
                strategy="bull_call_spread",
                score=80.0,
                rationale="candidate rebuilt from live data",
                legs_json=[],
                suggestion_json={},
            )
        )

    selected = ScoredStrategy(
        strategy_name="bull_call_spread",
        score=80.0,
        factors=FactorBreakdown(0.7, 0.6, 0.8, 0.9, 1.0, 0.5),
        suggestion=StrategySuggestion(
            strategy_name="bull_call_spread",
            legs=(),
            credit_or_debit=-50.0,
            max_loss=100.0,
            max_profit=50.0,
            prob_profit=0.45,
            suggested_quantity=1,
            defined_risk=True,
            rationale="candidate rebuilt from live data",
            reward_risk=0.5,
            expected_value=-1.0,
        ),
        rationale="candidate rebuilt from live data",
    )
    scan_result = SimpleNamespace(snapshot_id=snapshot_id, scored=(selected,))
    settings = SimpleNamespace(
        screener=SimpleNamespace(universe=["SPY"]),
        execution=SimpleNamespace(
            paper_only=True,
            zero_dte_only=True,
            zero_dte_entry_cutoff_minutes=30,
            max_single_trade_risk_pct=0.10,
        ),
        ibkr=SimpleNamespace(paper=True),
        scan=SimpleNamespace(scan_symbol_timeout_s=5.0, score_threshold=70.0),
    )
    context = cast(
        "DaemonContext",
        SimpleNamespace(
            engine=mcp_engine,
            settings=settings,
            ibkr=object(),
            ibkr_lock=asyncio.Lock(),
            resolver=None,
        ),
    )
    account = SimpleNamespace(net_liquidation_usd=10_000.0)
    positions_client = SimpleNamespace(get_account_summary=AsyncMock(return_value=account))
    payload = {
        "proposed_at": now.isoformat(),
        "symbol": "SPY",
        "direction": "bull",
        "iv_regime": "neutral",
        "strategy": "auto",
        "confidence": 0.8,
        "sources": ["FRED macro snapshot", "Finnhub SPY news"],
        "thesis": "Bounded paper hypothesis.",
        "checks": {
            "bot_health": True,
            "regime_history": True,
            "catalysts": True,
        },
    }

    with (
        patch(
            "optionsbot.daemon.market_hours.is_market_open",
            return_value=True,
        ),
        patch(
            "optionsbot.daemon.market_hours.minutes_to_nyse_close",
            return_value=120.0,
        ),
        patch(
            "optionsbot.scan.scan_symbol",
            new=AsyncMock(return_value=scan_result),
        ),
        patch(
            "optionsbot.ibkr.positions.PositionsClient",
            return_value=positions_client,
        ),
        patch(
            "optionsbot.daemon.scan_runner.rank_alert_candidates",
            return_value=[],
        ),
        patch(
            "optionsbot.daemon.candidate_evidence.capture_candidate_evidence",
            new=AsyncMock(return_value={"ready": True}),
        ),
    ):
        result = await _consume_entry_proposal(context, payload)

    assert result.startswith("Hermes proposal declined as score #")
    assert "non_positive_edge" in result
    with mcp_engine.connect() as conn:
        assert conn.execute(select(entry_reviews.c.id)).fetchall() == []


@pytest.mark.asyncio
async def test_entry_proposal_uses_fresh_economics_after_preliminary_rejection(
    mcp_engine: Engine,
) -> None:
    now = datetime.now(UTC)
    with mcp_engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=now,
                    spot=500.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_call_spread",
                    score=80.0,
                    rationale="candidate rebuilt from live data",
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(alerts).values(
                strategy_score_id=score_id,
                ts=now,
                symbol="SPY",
                strategy="bull_call_spread",
                score=80.0,
                status="sent",
                sent_ts=now,
                telegram_msg_id=12345,
            )
        )

    selected = ScoredStrategy(
        strategy_name="bull_call_spread",
        score=80.0,
        factors=FactorBreakdown(0.7, 0.6, 0.8, 0.9, 1.0, 0.5),
        suggestion=StrategySuggestion(
            strategy_name="bull_call_spread",
            legs=(),
            credit_or_debit=-150.0,
            max_loss=None,
            max_profit=None,
            prob_profit=0.45,
            suggested_quantity=1,
            defined_risk=True,
            rationale="preliminary scan economics",
            expected_value=-10.0,
        ),
        rationale="candidate rebuilt from live data",
    )
    scan_result = SimpleNamespace(snapshot_id=snapshot_id, scored=(selected,))
    settings = SimpleNamespace(
        screener=SimpleNamespace(universe=["SPY"]),
        execution=SimpleNamespace(
            paper_only=True,
            zero_dte_only=True,
            zero_dte_entry_cutoff_minutes=30,
            max_single_trade_risk_pct=0.10,
        ),
        ibkr=SimpleNamespace(paper=True),
        scan=SimpleNamespace(scan_symbol_timeout_s=5.0, score_threshold=70.0),
    )
    context = cast(
        "DaemonContext",
        SimpleNamespace(
            engine=mcp_engine,
            settings=settings,
            ibkr=object(),
            ibkr_lock=asyncio.Lock(),
            resolver=None,
        ),
    )
    positions_client = SimpleNamespace(
        get_account_summary=AsyncMock(return_value=SimpleNamespace(net_liquidation_usd=10_000.0))
    )
    payload = {
        "proposed_at": now.isoformat(),
        "symbol": "SPY",
        "direction": "bull",
        "iv_regime": "neutral",
        "strategy": "auto",
        "confidence": 0.8,
        "sources": ["FRED macro snapshot", "Finnhub SPY news"],
        "thesis": "Bounded paper hypothesis.",
        "checks": {
            "bot_health": True,
            "regime_history": True,
            "catalysts": True,
        },
    }
    evidence = {
        "ready": True,
        "economics": {
            "credit_or_debit": -40.0,
            "max_loss": 40.0,
            "max_profit": 60.0,
            "reward_risk": 1.5,
            "expected_value": 15.0,
        },
    }

    with (
        patch(
            "optionsbot.daemon.market_hours.is_market_open",
            return_value=True,
        ),
        patch(
            "optionsbot.daemon.market_hours.minutes_to_nyse_close",
            return_value=120.0,
        ),
        patch(
            "optionsbot.scan.scan_symbol",
            new=AsyncMock(return_value=scan_result),
        ),
        patch(
            "optionsbot.ibkr.positions.PositionsClient",
            return_value=positions_client,
        ),
        patch(
            "optionsbot.daemon.scan_runner.rank_alert_candidates",
            return_value=[],
        ),
        patch(
            "optionsbot.daemon.candidate_evidence.capture_candidate_evidence",
            new=AsyncMock(return_value=evidence),
        ),
        patch(
            "optionsbot.daemon.alert_pipeline.enqueue_alert",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "optionsbot.daemon.auto_executor.auto_execute_candidates",
            new=AsyncMock(return_value=0),
        ),
    ):
        result = await _consume_entry_proposal(context, payload)

    assert "OptionsBot submitted=False" in result
    with mcp_engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.strategy_score_id == score_id
    assert review.verdict == "vetted_paper_candidate"
    assert review.status == "held"


def test_halt_is_queued_then_monotonically_consumed_by_daemon(
    mcp_engine: Engine, tmp_path: Path
) -> None:
    intent_path = tmp_path / "intents.db"
    intent_engine = create_intent_engine(intent_path)
    context = RestrictedServerContext(
        engine=mcp_engine,
        intent_engine=intent_engine,
        max_pick_age_minutes=20,
    )
    halt = get_tools(nightwatch.register)["halt"]
    result = halt("boundary drill", "HALT_OPTIONSBOT", FakeCtx(context))
    assert result["ok"] is True
    assert result["killed"] == "pending_daemon_consumption"
    assert load_state(mcp_engine).killed is False

    daemon_context = cast("DaemonContext", SimpleNamespace(engine=mcp_engine))
    assert consume_control_intents(daemon_context, intent_path) == 1
    assert load_state(mcp_engine).killed is True
    assert load_state(mcp_engine).reason == "boundary drill"
    with intent_engine.connect() as conn:
        row = conn.execute(select(control_intents)).one()
    assert row.status == "processed"

    # Consumption is idempotent; a second pass cannot clear or duplicate it.
    assert consume_control_intents(daemon_context, intent_path) == 0
    assert load_state(mcp_engine).killed is True
