from __future__ import annotations

from datetime import UTC, datetime

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
