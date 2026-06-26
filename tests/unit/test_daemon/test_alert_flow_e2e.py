"""Daemon alert path: configured-threshold gating + full-tick e2e (IBK-92)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert, select

from optionsbot.analysis.types import MarketView
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.scan_runner import run_scan_tick
from optionsbot.scan.types import ScanResult
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.storage.schema import alerts, scan_runs, snapshots, watchlist


def _scored(name: str = "iron_condor", score: float = 85.0) -> ScoredStrategy:
    """A ScoredStrategy whose MagicMock suggestion carries REAL numeric fields
    so format_alert_markdown's f-string formatting works (legs=() renders an
    empty leg block, which is valid MarkdownV2)."""
    sug = MagicMock()
    sug.legs = ()
    sug.credit_or_debit = 1.25
    sug.max_loss = 3.75
    sug.max_profit = 1.25
    sug.prob_profit = 0.68
    sug.suggested_quantity = 5
    sug.defined_risk = True
    sug.reward_risk = 1.5
    sug.expected_value = 50.0
    sug.risk_tier = "balanced"
    return ScoredStrategy(
        strategy_name=name,
        score=score,
        factors=FactorBreakdown(0.7, 0.6, 0.8, 0.9, 1.0, 0.5),
        suggestion=sug,
        rationale="test rationale",
    )


def _mock_positions(net_liq: float = 5000.0) -> MagicMock:
    """A PositionsClient whose get_account_summary returns a USD net-liq, so the
    IBK-146 affordability gate in run_scan_tick passes (without it, account_value
    is None and rank_alert_candidates fail-closes, dropping every pick)."""
    from decimal import Decimal

    from optionsbot.ibkr.types import AccountSummary

    summary = AccountSummary(
        net_liquidation=Decimal(str(net_liq)),
        buying_power=None,
        available_funds=Decimal(str(net_liq)),
        currency="USD",
    )
    pos = MagicMock()
    pos.get_account_summary = AsyncMock(return_value=summary)
    return pos


def _scan_result(
    *, scored: tuple[ScoredStrategy, ...], snapshot_id: int = 1, symbol: str = "SPY"
) -> ScanResult:
    view = MarketView(
        direction="neutral",
        direction_strength="weak",
        iv_regime="high",
        iv_rank_value=0.7,
        earnings_in_window=False,
        warming_up=False,
    )
    return ScanResult(
        symbol=symbol,
        snapshot_id=snapshot_id,
        snapshot_ts=datetime(2026, 6, 2, 15, 30, tzinfo=UTC),
        view=view,
        scored=scored,
    )


def _seed_snapshot(daemon_context: DaemonContext, *, symbol: str = "SPY") -> int:
    """Insert a schema-valid snapshots row; dispatch_alert reads it to build
    the MarketView. Returns the new snapshot id."""
    with daemon_context.engine.begin() as conn:
        result = conn.execute(
            insert(snapshots).values(
                symbol=symbol,
                ts=datetime.now(UTC),
                spot=400.0,
                regime_dir="neutral",
                regime_iv="high",
            )
        )
        return int(result.inserted_primary_key[0])


def _add_watchlist(daemon_context: DaemonContext, symbol: str = "SPY") -> None:
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol=symbol, added_at=datetime.now(UTC)))


async def test_daemon_skips_score_below_configured_threshold(
    daemon_context: DaemonContext,
) -> None:
    """A score below scan.score_threshold must NOT be enqueued, even though it
    clears the hardcoded scoring DEFAULT_THRESHOLD (70). enqueue_alert is mocked
    so the assertion isolates the top_k threshold decision."""
    daemon_context.settings.scan.score_threshold = 95
    _add_watchlist(daemon_context)
    result = _scan_result(scored=(_scored("iron_condor", 85.0),))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), patch(
        "optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(return_value=result)
    ), patch(
        "optionsbot.daemon.scan_runner.enqueue_alert", new=AsyncMock()
    ) as mock_enqueue:
        summary = await run_scan_tick(daemon_context)

    assert mock_enqueue.await_count == 0
    assert summary.alerts_enqueued == 0


async def test_daemon_enqueues_score_at_configured_threshold(
    daemon_context: DaemonContext,
) -> None:
    """A score >= the configured threshold IS enqueued."""
    daemon_context.settings.scan.score_threshold = 80
    _add_watchlist(daemon_context)
    result = _scan_result(scored=(_scored("iron_condor", 85.0),))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), patch(
        "optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(return_value=result)
    ), patch(
        "optionsbot.daemon.scan_runner.PositionsClient", return_value=_mock_positions()
    ), patch(
        "optionsbot.daemon.scan_runner.enqueue_alert", new=AsyncMock(return_value=True)
    ) as mock_enqueue:
        summary = await run_scan_tick(daemon_context)

    assert mock_enqueue.await_count == 1
    assert summary.alerts_enqueued == 1


async def test_full_tick_dispatches_alert_end_to_end(
    daemon_context: DaemonContext,
) -> None:
    """run_scan_tick -> REAL enqueue_alert/dispatch_alert/dedup/formatter ->
    (fake) Telegram send. Proves the vertical wiring the per-layer tests skip.
    Only telegram is faked (mock_telegram fixture returns msg_id 12345)."""
    daemon_context.settings.scan.score_threshold = 70
    snap_id = _seed_snapshot(daemon_context)
    _add_watchlist(daemon_context)
    result = _scan_result(scored=(_scored("iron_condor", 85.0),), snapshot_id=snap_id)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), patch(
        "optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(return_value=result)
    ), patch(
        "optionsbot.daemon.scan_runner.PositionsClient", return_value=_mock_positions()
    ):
        summary = await run_scan_tick(daemon_context)

    # A rendered Telegram message was sent.
    daemon_context.telegram.send_message.assert_awaited_once()
    sent_text = daemon_context.telegram.send_message.await_args.args[0]
    assert "SPY" in sent_text
    assert "iron_condor" in sent_text
    # The view line is rendered from the seeded snapshot's regime fields, so its
    # presence proves dispatch_alert loaded the correct snapshot (not a default).
    assert "neutral/high" in sent_text

    # The alerts row was marked sent with the mock msg id, and counters agree.
    with daemon_context.engine.connect() as conn:
        alert_rows = conn.execute(select(alerts)).fetchall()
        run_rows = conn.execute(select(scan_runs)).fetchall()
    assert len(alert_rows) == 1
    assert alert_rows[0].symbol == "SPY"
    assert alert_rows[0].strategy == "iron_condor"
    assert alert_rows[0].status == "sent"
    assert alert_rows[0].telegram_msg_id == 12345
    assert alert_rows[0].sent_ts is not None
    assert summary.alerts_enqueued == 1
    assert len(run_rows) == 1
    assert run_rows[0].alerts_fired == 1


async def test_full_tick_dedup_suppresses_duplicate_second_tick(
    daemon_context: DaemonContext,
) -> None:
    """Two identical ticks: the first sends, the second is deduped by
    should_alert (same symbol/strategy, within cooldown, score unchanged so not
    > rescore delta). No duplicate send, no duplicate alerts row."""
    daemon_context.settings.scan.score_threshold = 70
    snap_id = _seed_snapshot(daemon_context)
    _add_watchlist(daemon_context)
    result = _scan_result(scored=(_scored("iron_condor", 85.0),), snapshot_id=snap_id)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), patch(
        "optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(return_value=result)
    ), patch(
        "optionsbot.daemon.scan_runner.PositionsClient", return_value=_mock_positions()
    ):
        first = await run_scan_tick(daemon_context)
        second = await run_scan_tick(daemon_context)

    daemon_context.telegram.send_message.assert_awaited_once()  # only the first tick
    assert first.alerts_enqueued == 1
    assert second.alerts_enqueued == 0
    with daemon_context.engine.connect() as conn:
        alert_rows = conn.execute(select(alerts)).fetchall()
        run_rows = conn.execute(select(scan_runs).order_by(scan_runs.c.id)).fetchall()
    assert len(alert_rows) == 1  # no duplicate row
    # Both ticks persist a scan_runs heartbeat; only the first fired an alert.
    assert len(run_rows) == 2
    assert run_rows[0].alerts_fired == 1
    assert run_rows[1].alerts_fired == 0
