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


async def test_watchlist_list_and_add_and_remove(daemon_context: DaemonContext) -> None:
    [empty] = await dispatch(daemon_context, "/watchlist list")
    assert "empty" in empty.text.lower()

    # add validates via the resolver (mock it so no IBKR needed)
    daemon_context.resolver.stock = AsyncMock(return_value=MagicMock())
    [added] = await dispatch(daemon_context, "/watchlist add aapl")
    assert "AAPL" in added.text
    [listed] = await dispatch(daemon_context, "/watchlist list")
    assert "AAPL" in listed.text

    [removed] = await dispatch(daemon_context, "/watchlist remove AAPL")
    assert "AAPL" in removed.text
    [empty2] = await dispatch(daemon_context, "/watchlist list")
    assert "empty" in empty2.text.lower()


async def test_watchlist_add_requires_symbol(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "/watchlist add")
    assert "usage" in reply.text.lower()


async def test_scan_orders_picks_by_edge(daemon_context: DaemonContext) -> None:
    """/scan leads with the highest risk-normalized-edge pick, not chain order."""
    from optionsbot.analysis.types import MarketView
    from optionsbot.scan.types import ScanResult
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    def _mk(name: str, rne: float) -> ScoredStrategy:
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
        sug.risk_normalized_expectancy = rne
        return ScoredStrategy(name, 80.0, FactorBreakdown(.5,.5,.5,.5,.5,.5), sug, "ok")

    view = MarketView("neutral", "weak", "high", 0.7, False, False)
    # Chain order puts the LOW-edge pick first; edge ranking must reorder.
    scored = (_mk("low_edge", 0.01), _mk("high_edge", 0.40), _mk("mid_edge", 0.10))
    result = ScanResult("SPY", 1, datetime(2026, 6, 5, 15, 30, tzinfo=UTC), view, scored)

    with patch("optionsbot.daemon.commands.scan_symbol", new=AsyncMock(return_value=result)):
        replies = await dispatch(daemon_context, "/scan spy")
    # First reply (top pick) is the highest-edge strategy.
    assert "high_edge" in replies[0].text


async def test_scan_warns_and_orders_when_no_positive_edge(
    daemon_context: DaemonContext,
) -> None:
    """All picks negative-EV: /scan prepends the no-edge banner and orders the
    losers by raw EV (least loss first), NOT by EV/max_loss."""
    from optionsbot.analysis.types import MarketView
    from optionsbot.scan.types import ScanResult
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    def _mk(name: str, ev: float, max_loss: float) -> ScoredStrategy:
        sug = MagicMock()
        sug.legs = ()
        sug.defined_risk = True
        sug.credit_or_debit = 1.0
        sug.max_loss = max_loss
        sug.prob_profit = 0.6
        sug.reward_risk = 1.0
        sug.expected_value = ev
        sug.risk_tier = "balanced"
        sug.suggested_quantity = 1
        sug.risk_normalized_expectancy = ev / max_loss
        return ScoredStrategy(name, 80.0, FactorBreakdown(.5, .5, .5, .5, .5, .5), sug, "ok")

    view = MarketView("neutral", "weak", "high", 0.7, False, False)
    csp = _mk("cash_secured_put", ev=-83.0, max_loss=19397.0)   # wins under EV/max_loss
    spread = _mk("bull_put_spread", ev=-49.0, max_loss=737.0)   # wins under raw EV
    result = ScanResult("NVDA", 1, datetime(2026, 6, 6, 15, 30, tzinfo=UTC), view, (csp, spread))

    with patch("optionsbot.daemon.commands.scan_symbol", new=AsyncMock(return_value=result)):
        replies = await dispatch(daemon_context, "/scan nvda")

    # First reply is the plain-text no-edge banner.
    assert replies[0].parse_mode is None
    assert "No positive-edge" in replies[0].text
    # The spread (less-negative EV) is the first PICK, ahead of the CSP.
    picks = [r.text for r in replies if r.parse_mode == "MarkdownV2"]
    assert "bull_put_spread" in picks[0]
    assert "cash_secured_put" in picks[1]


async def test_kill_trips_persisted_switch(daemon_context: DaemonContext) -> None:
    from optionsbot.execution.state import load_state

    [reply] = await dispatch(daemon_context, "/kill max pain today")
    assert "kill" in reply.text.lower()
    assert "max pain today" in reply.text
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert state.reason == "max pain today"


async def test_kill_without_args_uses_default_reason(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.execution.state import load_state

    await dispatch(daemon_context, "/kill")
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert state.reason  # some non-empty default


async def test_arm_clears_kill(daemon_context: DaemonContext) -> None:
    from optionsbot.execution.state import load_state

    await dispatch(daemon_context, "/kill oops")
    [reply] = await dispatch(daemon_context, "/arm")
    assert "clear" in reply.text.lower()
    assert load_state(daemon_context.engine).killed is False


async def test_exec_reports_disabled_by_default(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "/exec")
    assert "enabled: false" in reply.text
    assert "kill switch: clear" in reply.text
    assert "verdict:" in reply.text


async def test_exec_reports_tripped_kill_switch(daemon_context: DaemonContext) -> None:
    await dispatch(daemon_context, "/kill drawdown")
    [reply] = await dispatch(daemon_context, "/exec")
    assert "drawdown" in reply.text


async def test_help_lists_execution_commands(daemon_context: DaemonContext) -> None:
    [reply] = await dispatch(daemon_context, "/help")
    assert "/exec" in reply.text
    assert "/kill" in reply.text
    assert "/arm" in reply.text
