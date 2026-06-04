from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert

from optionsbot.daemon.commands import dispatch
from optionsbot.daemon.context import DaemonContext
from optionsbot.storage.schema import alerts, scan_runs, watchlist


async def test_help_lists_commands(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "/help")
    assert "/status" in reply.text and "/scan" in reply.text
    assert reply.parse_mode is None


async def test_unknown_command(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "/frobnicate")
    assert "unknown command" in reply.text.lower()


async def test_non_command_text_hints_help(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "hello")
    assert "/help" in reply.text


async def test_status_reports_state(daemon_context: DaemonContext) -> None:
    daemon_context.ibkr.is_connected = True  # MagicMock attr
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
        conn.execute(insert(scan_runs).values(
            started=datetime(2026, 6, 4, 15, 56, tzinfo=UTC),
            finished=datetime(2026, 6, 4, 15, 59, tzinfo=UTC),
            tickers_scanned=7, alerts_fired=3,
        ))
    [reply] = await dispatch(daemon_context, "/status")
    assert "scanned 7" in reply.text and "alerts 3" in reply.text
    assert "1 symbol" in reply.text
    assert "alerting: on" in reply.text


async def test_pause_resume_toggle_flag(daemon_context: DaemonContext) -> None:
    await dispatch(daemon_context, "/pause")
    assert daemon_context.alerting_paused is True
    await dispatch(daemon_context, "/resume")
    assert daemon_context.alerting_paused is False


async def test_last_lists_recent_alerts(daemon_context: DaemonContext) -> None:
    with daemon_context.engine.begin() as conn:
        for i, sym in enumerate(("SPY", "AAPL")):
            conn.execute(insert(alerts).values(
                ts=datetime(2026, 6, 4, 12, i, tzinfo=UTC), symbol=sym,
                strategy="iron_condor", score=80.0 + i, status="sent",
            ))
    [reply] = await dispatch(daemon_context, "/last 5")
    assert "SPY" in reply.text and "AAPL" in reply.text


async def test_scan_returns_formatted_picks(daemon_context: DaemonContext) -> None:
    from optionsbot.analysis.types import MarketView
    from optionsbot.scan.types import ScanResult
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    sug = MagicMock()
    sug.legs = ()
    sug.defined_risk = True
    sug.credit_or_debit = 1.0
    sug.max_loss = 2.0
    sug.prob_profit = 0.6
    sug.reward_risk = 1.0
    sug.expected_value = 5.0
    sug.risk_tier = "balanced"
    sug.suggested_quantity = 1
    scored = ScoredStrategy("iron_condor", 88.0, FactorBreakdown(.5,.5,.5,.5,.5,.5), sug, "ok")
    view = MarketView("neutral", "weak", "high", 0.7, False, False)
    result = ScanResult("SPY", 1, datetime(2026, 6, 4, 15, 30, tzinfo=UTC), view, (scored,))

    with patch("optionsbot.daemon.commands.scan_symbol", new=AsyncMock(return_value=result)):
        replies = await dispatch(daemon_context, "/scan spy")
    assert any("iron_condor" in r.text for r in replies)
    assert all(r.parse_mode == "MarkdownV2" for r in replies)


async def test_scan_no_symbol_usage(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "/scan")
    assert "usage" in reply.text.lower()


async def test_screen_lists_candidates(daemon_context: DaemonContext) -> None:
    from optionsbot.screener.screen import ScreenCandidate
    cands = (ScreenCandidate("NVDA", 0.9, 2e9), ScreenCandidate("AMD", 0.7, 5e8))
    with patch("optionsbot.daemon.commands.screen_universe", new=AsyncMock(return_value=cands)):
        [reply] = await dispatch(daemon_context, "/screen 2")
    assert "NVDA" in reply.text and "AMD" in reply.text
