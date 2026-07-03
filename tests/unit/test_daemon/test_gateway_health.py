"""Gateway wedge-detection + page-the-human (IBK-137 Increment 2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from optionsbot.config import MonitorSettings
from optionsbot.daemon.gateway_health import (
    GatewayHealthMonitor,
    count_budget_timeouts,
    page_gateway_health,
)

_NOW = datetime(2026, 7, 6, 15, 0, tzinfo=UTC)  # a Monday, mid-session


def _settings(**kw) -> MonitorSettings:
    return MonitorSettings(**kw)


def _eval(m, **kw):
    defaults = dict(
        now=_NOW, market_open=True, connected=True,
        tickers_scanned=10, budget_timeouts=0, open_positions=2,
        settings=_settings(),
    )
    defaults.update(kw)
    return m.evaluate(**defaults)


# --- config -----------------------------------------------------------------


def test_monitor_settings_defaults() -> None:
    s = MonitorSettings()
    assert s.enabled is True
    assert s.wedge_min_budget_timeouts == 3
    assert s.page_repeat_minutes == 30


# --- error-string parsing ----------------------------------------------------


def test_count_budget_timeouts_parses_scan_runner_error_strings() -> None:
    errors = [
        "SPY: TimeoutError (scan budget)",     # the exact scan_runner format
        "QQQ: TimeoutError (scan budget)",
        "BORK: ValueError: chain fetch failed",  # non-budget error -> not counted
    ]
    assert count_budget_timeouts(errors) == 2
    assert count_budget_timeouts([]) == 0


# --- wedge condition ----------------------------------------------------------


def test_wedge_majority_timeouts_pages_once() -> None:
    m = GatewayHealthMonitor()
    msgs = _eval(m, tickers_scanned=2, budget_timeouts=15)
    assert len(msgs) == 1
    assert "WEDGED" in msgs[0]
    assert "15" in msgs[0]  # timeout count named in the page


def test_wedge_minority_timeouts_do_not_page() -> None:
    m = GatewayHealthMonitor()
    # 3 timeouts meets the floor but is NOT a majority (10 scanned fine).
    assert _eval(m, tickers_scanned=10, budget_timeouts=3) == []


def test_wedge_below_floor_does_not_page() -> None:
    m = GatewayHealthMonitor()
    # majority (2 > 1) but under the min floor of 3 -> tiny scan, no page
    assert _eval(m, tickers_scanned=1, budget_timeouts=2) == []


def test_wedge_market_closed_does_not_page() -> None:
    m = GatewayHealthMonitor()
    assert _eval(m, market_open=False, tickers_scanned=0, budget_timeouts=20) == []


# --- disconnect condition ------------------------------------------------------


def test_disconnected_with_open_positions_pages() -> None:
    m = GatewayHealthMonitor()
    msgs = _eval(m, connected=False, tickers_scanned=0, open_positions=2)
    assert len(msgs) == 1
    assert "DISCONNECTED" in msgs[0]
    assert "2" in msgs[0]  # open-position count named


def test_disconnected_without_positions_does_not_page() -> None:
    m = GatewayHealthMonitor()
    assert _eval(m, connected=False, tickers_scanned=0, open_positions=0) == []


# --- persist / re-page / clear --------------------------------------------------


def test_persisting_wedge_repages_only_after_interval() -> None:
    m = GatewayHealthMonitor()
    assert len(_eval(m, tickers_scanned=0, budget_timeouts=10)) == 1  # ENTER
    # 15 min later (inside the 30-min window): silent
    t2 = _NOW + timedelta(minutes=15)
    assert _eval(m, now=t2, tickers_scanned=0, budget_timeouts=10) == []
    # 31 min after the first page: re-page
    t3 = _NOW + timedelta(minutes=31)
    msgs = _eval(m, now=t3, tickers_scanned=0, budget_timeouts=10)
    assert len(msgs) == 1
    assert "WEDGED" in msgs[0]


def test_recovery_sends_one_recovered_message_and_resets() -> None:
    m = GatewayHealthMonitor()
    _eval(m, tickers_scanned=0, budget_timeouts=10)  # ENTER
    t2 = _NOW + timedelta(minutes=15)
    msgs = _eval(m, now=t2, tickers_scanned=12, budget_timeouts=0)  # healthy again
    assert len(msgs) == 1
    assert "recovered" in msgs[0].lower()
    # fully reset: staying healthy stays silent; a NEW wedge pages immediately
    t3 = _NOW + timedelta(minutes=16)
    assert _eval(m, now=t3, tickers_scanned=12, budget_timeouts=0) == []
    t4 = _NOW + timedelta(minutes=17)
    assert len(_eval(m, now=t4, tickers_scanned=0, budget_timeouts=10)) == 1


def test_reason_change_pages_immediately_even_inside_repage_window() -> None:
    """WEDGED -> DISCONNECTED is a severity ESCALATION (protection degraded ->
    DOWN): the changed condition must page NOW, not wait out the 30-min window."""
    m = GatewayHealthMonitor()
    assert len(_eval(m, tickers_scanned=0, budget_timeouts=10)) == 1  # ENTER wedged
    t2 = _NOW + timedelta(minutes=5)  # well inside the 30-min re-page window
    msgs = _eval(
        m, now=t2, connected=False, tickers_scanned=0, budget_timeouts=0,
        open_positions=2,
    )
    assert len(msgs) == 1
    assert "DISCONNECTED" in msgs[0]


def test_disabled_monitor_never_pages() -> None:
    m = GatewayHealthMonitor()
    assert _eval(
        m, tickers_scanned=0, budget_timeouts=20, settings=_settings(enabled=False)
    ) == []


# --- async wiring seam -----------------------------------------------------------


def _summary(tickers_scanned: int, errors: list[str]):
    s = MagicMock()
    s.tickers_scanned = tickers_scanned
    s.errors = errors
    return s


async def test_page_gateway_health_sends_monitor_messages() -> None:
    context = MagicMock()
    context.monitor = GatewayHealthMonitor()
    context.ibkr.is_connected = True
    context.telegram.send_message = AsyncMock()
    context.settings.monitor = _settings()
    summary = _summary(0, [f"S{i}: TimeoutError (scan budget)" for i in range(10)])

    with patch(
        "optionsbot.daemon.gateway_health.is_market_open", return_value=True
    ), patch(
        "optionsbot.daemon.gateway_health._open_entries", return_value=[MagicMock()]
    ):
        await page_gateway_health(context, summary, now=_NOW)

    context.telegram.send_message.assert_awaited_once()
    assert "WEDGED" in context.telegram.send_message.await_args.args[0]


async def test_page_gateway_health_never_raises() -> None:
    """A telegram (or any) failure inside the helper is swallowed -- health
    paging must not poison the scan tick."""
    context = MagicMock()
    context.monitor = GatewayHealthMonitor()
    context.ibkr.is_connected = True
    context.telegram.send_message = AsyncMock(side_effect=RuntimeError("tg down"))
    context.settings.monitor = _settings()
    summary = _summary(0, [f"S{i}: TimeoutError (scan budget)" for i in range(10)])

    with patch(
        "optionsbot.daemon.gateway_health.is_market_open", return_value=True
    ), patch(
        "optionsbot.daemon.gateway_health._open_entries", return_value=[]
    ):
        await page_gateway_health(context, summary, now=_NOW)  # must not raise
