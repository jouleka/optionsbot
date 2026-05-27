"""Tests for alert dedup logic (IBK-62)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import insert

from optionsbot.config import Settings
from optionsbot.daemon.alert_dedup import should_alert
from optionsbot.storage.schema import alerts


def _insert_sent_alert(
    engine, symbol: str, strategy: str, *,
    ts: datetime, score: float,
) -> None:
    with engine.begin() as conn:
        conn.execute(insert(alerts).values(
            ts=ts, symbol=symbol, strategy=strategy, score=score,
            status="sent", sent_ts=ts,
        ))


def test_first_alert_for_pair_always_fires(daemon_engine, daemon_settings) -> None:
    assert should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 80.0)


def test_within_cooldown_with_no_score_delta_is_suppressed(
    daemon_engine, daemon_settings: Settings,
) -> None:
    daemon_settings.scan.alert_cooldown_hours = 4
    daemon_settings.scan.alert_rescore_delta = 10
    _insert_sent_alert(
        daemon_engine, "AAPL", "iron_condor",
        ts=datetime.now(UTC) - timedelta(hours=2),
        score=80.0,
    )
    assert not should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 80.0)


def test_within_cooldown_but_score_jumped_fires(daemon_engine, daemon_settings) -> None:
    daemon_settings.scan.alert_cooldown_hours = 4
    daemon_settings.scan.alert_rescore_delta = 10
    _insert_sent_alert(
        daemon_engine, "AAPL", "iron_condor",
        ts=datetime.now(UTC) - timedelta(hours=2),
        score=80.0,
    )
    assert should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 91.0)


def test_within_cooldown_with_borderline_delta_does_not_fire(
    daemon_engine, daemon_settings,
) -> None:
    """Delta of exactly rescore_delta does NOT trigger; must be strictly greater."""
    daemon_settings.scan.alert_cooldown_hours = 4
    daemon_settings.scan.alert_rescore_delta = 10
    _insert_sent_alert(
        daemon_engine, "AAPL", "iron_condor",
        ts=datetime.now(UTC) - timedelta(hours=2),
        score=80.0,
    )
    # 80 + 10 = 90 — equal, not strictly greater.
    assert not should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 90.0)


def test_past_cooldown_fires_regardless_of_score(daemon_engine, daemon_settings) -> None:
    daemon_settings.scan.alert_cooldown_hours = 4
    daemon_settings.scan.alert_rescore_delta = 10
    _insert_sent_alert(
        daemon_engine, "AAPL", "iron_condor",
        ts=datetime.now(UTC) - timedelta(hours=5),
        score=80.0,
    )
    # Same score, but past cooldown.
    assert should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 80.0)


def test_pending_or_failed_status_does_not_count_as_last_sent(
    daemon_engine, daemon_settings,
) -> None:
    """Only 'sent' alerts factor into dedup; pending/failed/dropped are ignored."""
    daemon_settings.scan.alert_cooldown_hours = 4
    daemon_settings.scan.alert_rescore_delta = 10
    with daemon_engine.begin() as conn:
        conn.execute(insert(alerts).values(
            ts=datetime.now(UTC) - timedelta(hours=1),
            symbol="AAPL", strategy="iron_condor", score=80.0,
            status="pending",
        ))
    # Pending alert exists but should_alert ignores it -> True.
    assert should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 80.0)


def test_dedup_is_per_symbol_strategy_pair(daemon_engine, daemon_settings) -> None:
    """Recent send for iron_condor on AAPL doesn't block iron_butterfly on AAPL."""
    daemon_settings.scan.alert_cooldown_hours = 4
    _insert_sent_alert(
        daemon_engine, "AAPL", "iron_condor",
        ts=datetime.now(UTC) - timedelta(hours=1),
        score=80.0,
    )
    assert should_alert(daemon_engine, daemon_settings, "AAPL", "iron_butterfly", 80.0)
    assert should_alert(daemon_engine, daemon_settings, "MSFT", "iron_condor", 80.0)


def test_cooldown_zero_disables_cooldown(daemon_engine, daemon_settings) -> None:
    """alert_cooldown_hours=0 effectively disables the cooldown gate."""
    daemon_settings.scan.alert_cooldown_hours = 0
    _insert_sent_alert(
        daemon_engine, "AAPL", "iron_condor",
        ts=datetime.now(UTC) - timedelta(seconds=5),
        score=80.0,
    )
    # Cooldown=0 → past cooldown immediately → fires.
    assert should_alert(daemon_engine, daemon_settings, "AAPL", "iron_condor", 80.0)
