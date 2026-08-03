"""Tests for IBK-138 nightwatch MCP tools."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import insert, select, update

from optionsbot.execution.state import load_state
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.nightwatch import _compact_learning_feedback, register
from optionsbot.storage.schema import (
    alerts,
    entry_reviews,
    exit_requests,
    orders,
    snapshots,
    strategy_scores,
)
from tests.unit.test_mcp.conftest import FakeCtx, get_tools

NOW = datetime.now(UTC)

LEGS = [
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
        "strike": 575.0,
        "right": "P",
        "quantity": 1,
    },
]


def _snapshot_with_score(
    server_context: ServerContext, *, score: float = 82.0, alerted: bool = True
) -> int:
    with server_context.engine.begin() as conn:
        snap_pk = conn.execute(
            insert(snapshots).values(
                symbol="SPY",
                ts=NOW,
                spot=550.0,
                iv_rank=0.62,
                hv20=0.18,
                iv_hv_ratio=1.4,
                expected_move=8.5,
                regime_dir="neutral",
                regime_iv="high",
                raw_json={
                    "earnings_in_window": False,
                    "relative_strength": 0.03,
                    "delayed": False,
                    "warming_up": False,
                },
            )
        ).inserted_primary_key
        snap_id = int(snap_pk[0])
        score_pk = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snap_id,
                strategy="bull_put_spread",
                score=score,
                rationale="positive expectancy, defined risk",
                legs_json=LEGS,
                suggestion_json={
                    "defined_risk": True,
                    "credit_or_debit": 120.0,
                    "max_loss": 380.0,
                    "max_profit": 120.0,
                    "prob_profit": 0.68,
                    "expected_value": 14.5,
                    "suggested_quantity": 1,
                },
            )
        ).inserted_primary_key
        score_id = int(score_pk[0])
        suggestion = {
            "defined_risk": True,
            "credit_or_debit": 120.0,
            "max_loss": 380.0,
            "max_profit": 120.0,
            "prob_profit": 0.68,
            "expected_value": 14.5,
            "suggested_quantity": 1,
            "review_evidence": {
                "schema_version": 2,
                "source": "trusted_daemon",
                "score_id": score_id,
                "captured_at": NOW.isoformat(),
                "ready": True,
                "readiness_issues": [],
                "option_quotes": [{"expiry": "20260717", "strike": 580.0}],
                "combo": {"spread_allowed": True},
                "candidate_greeks": {"complete": True},
                "exposure": {"complete": True},
                "expiration_assignment": {},
                "market_timing": {"entry_window_open": True},
                "account": {
                    "net_liquidation_usd": 100_000.0,
                    "buying_power": 100_000.0,
                    "available_funds": 100_000.0,
                },
                "risk": {
                    "execution_allowed": True,
                    "paper_only": True,
                    "entry_loss_guard_allowed": True,
                    "single_trade_risk_allowed": True,
                    "portfolio_heat_allowed": True,
                    "candidate_affordable": True,
                    "bp_deployment_allowed": True,
                },
            },
        }
        conn.execute(
            update(strategy_scores)
            .where(strategy_scores.c.id == score_id)
            .values(suggestion_json=suggestion)
        )
        if alerted:
            conn.execute(
                insert(alerts).values(
                    strategy_score_id=score_id,
                    ts=NOW,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    score=score,
                    status="sent",
                    sent_ts=NOW,
                    telegram_msg_id=12345,
                )
            )
    return score_id


def _alert_id_for_score(server_context: ServerContext, score_id: int) -> int:
    with server_context.engine.connect() as conn:
        return int(
            conn.execute(
                select(alerts.c.id).where(alerts.c.strategy_score_id == score_id)
            ).scalar_one()
        )


def test_pending_picks_returns_grounded_pre_trade_packet(server_context: ServerContext) -> None:
    score_id = _snapshot_with_score(server_context)
    pending_picks = get_tools(register)["pending_picks"]

    result = pending_picks(limit=5, min_score=70.0, max_age_minutes=60, ctx=FakeCtx(server_context))

    assert result["ok"] is True
    assert result["count"] == 1
    pick = result["picks"][0]
    assert pick["pick_id"] == score_id
    assert pick["symbol"] == "SPY"
    assert pick["strategy"] == "bull_put_spread"
    assert pick["suggestion"]["expected_value"] == 14.5
    assert pick["evidence_ready"] is True
    assert pick["full_packet_embedded"] is False
    assert pick["next_tool"] == "pick_review_packet"
    assert result["packet_tool"] == "pick_review_packet"
    assert result["learning_feedback"]["review_calls"] == 0
    assert len(json.dumps(result, default=str)) < 50_000


def test_pick_review_packet_returns_complete_exact_evidence(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    packet = get_tools(register)["pick_review_packet"](
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        ctx=FakeCtx(server_context),
    )

    assert packet["ok"] is True
    pick = packet["pick"]
    assert pick["pick_id"] == score_id
    assert pick["alert_id"] == _alert_id_for_score(server_context, score_id)
    assert pick["market"]["iv_rank"] == 0.62
    assert pick["market"]["relative_strength"] == 0.03
    assert pick["review_evidence"]["source"] == "trusted_daemon"
    assert pick["review_evidence"]["option_quotes"]
    assert pick["legs"] == LEGS
    assert "news/catalyst corroboration" in packet["rubric"]["must_check"]
    assert len(json.dumps(packet, default=str)) < 50_000


def test_pending_picks_omits_unalerted_scores(server_context: ServerContext) -> None:
    _snapshot_with_score(server_context, alerted=False)
    pending_picks = get_tools(register)["pending_picks"]

    result = pending_picks(
        limit=5,
        min_score=70.0,
        max_age_minutes=60,
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["picks"] == []


def test_pending_picks_ten_item_queue_stays_below_transport_budget(
    server_context: ServerContext,
) -> None:
    score_ids = [_snapshot_with_score(server_context, score=90.0 - index) for index in range(10)]

    result = get_tools(register)["pending_picks"](
        limit=10,
        min_score=50.0,
        max_age_minutes=60,
        ctx=FakeCtx(server_context),
    )

    assert result["count"] == 10
    assert {pick["pick_id"] for pick in result["picks"]} == set(score_ids)
    assert all(
        pick["full_packet_embedded"] is False and pick["next_tool"] == "pick_review_packet"
        for pick in result["picks"]
    )
    assert len(json.dumps(result, default=str)) < 50_000


def test_compact_feedback_keeps_only_relevant_exact_tuple_groups() -> None:
    groups = {
        "META|iron_condor": {"calls": 3},
        "SPY|bull_put_spread": {"calls": 4},
        "QQQ|iron_condor": {"calls": 5},
        "AAPL|long_call": {"calls": 6},
    }
    feedback = {
        name: {"calls": 18, "by_strategy_symbol": groups}
        for name in (
            "forecast_call_summary",
            "terminal_call_summary",
            "actual_trade_summary",
            "guarded_call_summary",
        )
    }
    feedback["forecast_call_summary"]["by_strategy_symbol_sessions"] = groups

    compact = _compact_learning_feedback(
        feedback,
        relevant_pairs={"META|iron_condor"},
        independent_symbols={"SPY", "QQQ", "IWM"},
    )

    expected = {
        "META|iron_condor",
        "SPY|bull_put_spread",
        "QQQ|iron_condor",
    }
    for name in feedback:
        assert set(compact[name]["by_strategy_symbol"]) == expected
        assert compact[name]["calls"] == 18
    assert set(compact["forecast_call_summary"]["by_strategy_symbol_sessions"]) == expected


def test_compact_feedback_bounds_verbose_recent_rows() -> None:
    feedback = {
        "recent_lessons": [
            {"review_id": index, "review_reason": "r" * 1_000} for index in range(10)
        ],
        "recent_proposal_decisions": [
            {
                "intent_id": index,
                "symbol": "SPY",
                "status": "processed",
                "decision": "d" * 1_000,
            }
            for index in range(10)
        ],
    }

    compact = _compact_learning_feedback(
        feedback,
        relevant_pairs=set(),
        independent_symbols={"SPY", "QQQ", "IWM"},
    )

    assert len(compact["recent_lessons"]) == 5
    assert len(compact["recent_lessons"][0]["review_reason_summary"]) == 240
    assert len(compact["recent_proposal_decisions"]) == 5
    assert len(compact["recent_proposal_decisions"][0]["decision_summary"]) == 300


def test_submit_entry_review_queues_complete_vetted_candidate(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]
    checks = {
        "bot_health": True,
        "candidate": True,
        "microstructure": True,
        "greeks": True,
        "regime_history": True,
        "catalysts": True,
        "account_risk": True,
    }

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["issuer calendar", "independent market-data source"],
        reason="All seven gates pass; daemon must still reprice and rerun deterministic gates.",
        checks=checks,
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True
    assert result["status"] == "requested"
    with server_context.engine.connect() as conn:
        row = conn.execute(select(entry_reviews)).one()
    assert row.strategy_score_id == score_id
    assert row.alert_id == _alert_id_for_score(server_context, score_id)
    assert row.verdict == "vetted_paper_candidate"
    assert row.checks_json == checks
    assert row.status == "requested"


def test_submit_entry_review_rejects_unalerted_score(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context, alerted=False)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=999_999,
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["issuer calendar", "independent market-data source"],
        reason="The evidence passes, but the bot did not select this candidate.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "pick_not_alerted"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_rejects_alert_for_another_pick(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    other_score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, other_score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["source A", "source B"],
        reason="The alert and candidate identities must be inseparable.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "alert_pick_mismatch"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_rejects_incomplete_vetted_checks(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["source A", "source B"],
        reason="Greeks were unavailable.",
        checks={
            "bot_health": True,
            "candidate": True,
            "microstructure": True,
            "greeks": False,
            "regime_history": True,
            "catalysts": True,
            "account_risk": True,
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "all_seven_checks_must_pass"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_rejects_delayed_candidate(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    with server_context.engine.begin() as conn:
        snapshot_id = conn.execute(
            select(strategy_scores.c.snapshot_id).where(strategy_scores.c.id == score_id)
        ).scalar_one()
        conn.execute(
            update(snapshots)
            .where(snapshots.c.id == snapshot_id)
            .values(raw_json={"delayed": True, "warming_up": False})
        )
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["source A", "source B"],
        reason="Claimed pass despite delayed snapshot.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "candidate_data_unready"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_accepts_explicit_hv_proxy_during_iv_warmup(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    with server_context.engine.begin() as conn:
        snapshot_id = conn.execute(
            select(strategy_scores.c.snapshot_id).where(strategy_scores.c.id == score_id)
        ).scalar_one()
        conn.execute(
            update(snapshots)
            .where(snapshots.c.id == snapshot_id)
            .values(
                raw_json={
                    "delayed": False,
                    "warming_up": True,
                    "iv_rank_is_proxy": True,
                }
            )
        )
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["source A", "source B"],
        reason="Live evidence passes and the explicit long-history HV proxy backs IV regime.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True


def test_submit_entry_review_rejects_missing_daemon_evidence(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    with server_context.engine.begin() as conn:
        suggestion = conn.execute(
            select(strategy_scores.c.suggestion_json).where(strategy_scores.c.id == score_id)
        ).scalar_one()
        suggestion = dict(suggestion)
        suggestion.pop("review_evidence")
        conn.execute(
            update(strategy_scores)
            .where(strategy_scores.c.id == score_id)
            .values(suggestion_json=suggestion)
        )

    result = get_tools(register)["submit_entry_review"](
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.91,
        sources=["source A", "source B"],
        reason="Claims pass without the daemon packet.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "candidate_evidence_unready"}


def test_submit_entry_review_rejects_low_confidence_vetted(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.79,
        sources=["source A", "source B"],
        reason="Evidence is not strong enough.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "confidence_below_threshold"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_requires_two_sources_for_vetted(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.90,
        sources=["one source only"],
        reason="Insufficient corroboration.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "two_sources_required"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_requires_two_distinct_sources_for_vetted(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.90,
        sources=["same source", "same source"],
        reason="Duplicating one source is not corroboration.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "two_distinct_sources_required"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_rejects_non_positive_candidate_economics(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    with server_context.engine.begin() as conn:
        conn.execute(
            update(strategy_scores)
            .where(strategy_scores.c.id == score_id)
            .values(
                suggestion_json={
                    "defined_risk": True,
                    "credit_or_debit": 120.0,
                    "max_loss": 380.0,
                    "max_profit": 120.0,
                    "prob_profit": 0.68,
                    "expected_value": -1.0,
                    "suggested_quantity": 1,
                }
            )
        )
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.90,
        sources=["source A", "source B"],
        reason="Hermes must not override negative persisted EV.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "candidate_not_positive_defined_risk"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_accepts_positive_ev_long_option_with_unbounded_profit(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    long_call_legs = [
        {
            "symbol": "SPY",
            "side": "buy",
            "sec_type": "OPT",
            "expiry": "20260717",
            "strike": 580.0,
            "right": "C",
            "quantity": 1,
        }
    ]
    with server_context.engine.begin() as conn:
        suggestion = conn.execute(
            select(strategy_scores.c.suggestion_json).where(
                strategy_scores.c.id == score_id
            )
        ).scalar_one()
        updated = dict(suggestion)
        updated.update(
            defined_risk=True,
            credit_or_debit=-150.0,
            max_loss=150.0,
            max_profit=None,
            prob_profit=0.45,
            expected_value=20.0,
        )
        conn.execute(
            update(strategy_scores)
            .where(strategy_scores.c.id == score_id)
            .values(
                strategy="long_call",
                legs_json=long_call_legs,
                suggestion_json=updated,
            )
        )
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.90,
        sources=["source A", "source B"],
        reason="Positive-EV long call has bounded loss and uncapped profit.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True
    assert result["status"] == "requested"


def test_submit_entry_review_rejects_stale_pick(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    with server_context.engine.begin() as conn:
        snapshot_id = conn.execute(
            select(strategy_scores.c.snapshot_id).where(strategy_scores.c.id == score_id)
        ).scalar_one()
        conn.execute(
            update(snapshots)
            .where(snapshots.c.id == snapshot_id)
            .values(ts=NOW - timedelta(minutes=31))
        )
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.90,
        sources=["source A", "source B"],
        reason="Stale candidates must not be queued.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "stale_pick"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_requires_reason(server_context: ServerContext) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="NO TRADE",
        confidence=0.90,
        sources=["source A", "source B"],
        reason="   ",
        checks={"microstructure": False},
        ctx=FakeCtx(server_context),
    )

    assert result == {"ok": False, "error": "reason_required"}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(entry_reviews)).first() is None


def test_submit_entry_review_persists_no_trade_as_non_executable(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]

    result = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="NO TRADE",
        confidence=0.97,
        sources=["fresh option chain", "issuer calendar"],
        reason="Option quotes are delayed and earnings risk is unresolved.",
        checks={"microstructure": False, "catalysts": False},
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True
    assert result["status"] == "refused"
    with server_context.engine.connect() as conn:
        row = conn.execute(select(entry_reviews)).one()
    assert row.verdict == "no_trade"
    assert row.status == "refused"


def test_submit_entry_review_does_not_override_existing_no_trade(
    server_context: ServerContext,
) -> None:
    score_id = _snapshot_with_score(server_context)
    submit_entry_review = get_tools(register)["submit_entry_review"]
    submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="NO TRADE",
        confidence=0.95,
        sources=["source A", "source B"],
        reason="Initial fail-closed verdict.",
        checks={"microstructure": False},
        ctx=FakeCtx(server_context),
    )

    second = submit_entry_review(
        pick_id=score_id,
        alert_id=_alert_id_for_score(server_context, score_id),
        verdict="VETTED PAPER CANDIDATE",
        confidence=0.95,
        sources=["source A", "source B"],
        reason="Must not replace the prior veto for this exact pick.",
        checks={
            name: True
            for name in (
                "bot_health",
                "candidate",
                "microstructure",
                "greeks",
                "regime_history",
                "catalysts",
                "account_risk",
            )
        },
        ctx=FakeCtx(server_context),
    )

    assert second["ok"] is True
    assert second["already_reviewed"] is True
    assert second["status"] == "refused"
    with server_context.engine.connect() as conn:
        rows = conn.execute(select(entry_reviews)).all()
    assert len(rows) == 1
    assert rows[0].verdict == "no_trade"


def test_request_exit_queues_valid_request_for_open_position(server_context: ServerContext) -> None:
    with server_context.engine.begin() as conn:
        pk = conn.execute(
            insert(orders).values(
                intent="open",
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=LEGS,
                quantity=1,
                status="filled",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                ib_order_id=11,
                order_ref="obot-1",
                reprice_count=0,
            )
        ).inserted_primary_key
    position_id = int(pk[0])
    request_exit = get_tools(register)["request_exit"]

    result = request_exit(
        position_id=position_id,
        catalyst_type="downgrade_upgrade",
        confidence=0.82,
        sources=["Reuters headline", "price/volume corroboration"],
        reason="Downgrade plus adverse tape; asking daemon gate to evaluate close-only exit.",
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True
    assert result["status"] == "requested"
    with server_context.engine.connect() as conn:
        row = conn.execute(select(exit_requests)).one()
    assert row.position_id == position_id
    assert row.catalyst_type == "downgrade_upgrade"
    assert row.confidence == 0.82
    assert row.sources_json == ["Reuters headline", "price/volume corroboration"]
    assert row.status == "requested"


def test_request_exit_refuses_unknown_position(server_context: ServerContext) -> None:
    request_exit = get_tools(register)["request_exit"]

    result = request_exit(
        position_id=999,
        catalyst_type="downgrade_upgrade",
        confidence=0.9,
        sources=["source A", "source B"],
        reason="test",
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is False
    assert result["error"] == "position_not_open"


def test_halt_requires_exact_confirmation(server_context: ServerContext) -> None:
    halt = get_tools(register)["halt"]

    refused = halt(reason="test", confirm="wrong", ctx=FakeCtx(server_context))
    assert refused["ok"] is False
    assert refused["error"] == "confirmation_required"
    assert load_state(server_context.engine).killed is False

    accepted = halt(
        reason="IBK-138 smoke test halt", confirm="HALT_OPTIONSBOT", ctx=FakeCtx(server_context)
    )
    assert accepted["ok"] is True
    assert accepted["killed"] is True
    assert load_state(server_context.engine).killed is True
