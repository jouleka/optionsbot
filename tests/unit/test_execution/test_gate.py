"""Truth-table tests for the execution arming gate (IBK-123)."""

from __future__ import annotations

from optionsbot.config import Settings
from optionsbot.execution.gate import can_execute
from optionsbot.execution.state import ExecutionState

NOT_KILLED = ExecutionState(killed=False, reason=None, ts=None)
KILLED = ExecutionState(killed=True, reason="manual /kill", ts=None)


def _settings(
    *,
    enabled: bool = False,
    paper_only: bool = True,
    paper: bool = True,
    port: int = 4002,
) -> Settings:
    s = Settings()
    s.execution.enabled = enabled
    s.execution.paper_only = paper_only
    s.ibkr.paper = paper
    s.ibkr.port = port
    return s


def test_default_settings_deny_with_disabled_reason() -> None:
    result = can_execute(Settings(), NOT_KILLED)
    assert result.allowed is False
    assert "enabled" in result.reason


def test_enabled_on_paper_gateway_port_allows() -> None:
    result = can_execute(_settings(enabled=True, port=4002), NOT_KILLED)
    assert result.allowed is True


def test_enabled_on_paper_tws_port_allows() -> None:
    result = can_execute(_settings(enabled=True, port=7497), NOT_KILLED)
    assert result.allowed is True


def test_live_account_flag_denies_even_when_enabled() -> None:
    result = can_execute(_settings(enabled=True, paper=False), NOT_KILLED)
    assert result.allowed is False
    assert "paper" in result.reason.lower()


def test_live_gateway_port_denies() -> None:
    result = can_execute(_settings(enabled=True, port=4001), NOT_KILLED)
    assert result.allowed is False
    assert "4001" in result.reason


def test_live_tws_port_denies() -> None:
    result = can_execute(_settings(enabled=True, port=7496), NOT_KILLED)
    assert result.allowed is False
    assert "7496" in result.reason


def test_kill_switch_denies_and_reports_reason() -> None:
    result = can_execute(_settings(enabled=True), KILLED)
    assert result.allowed is False
    assert "manual /kill" in result.reason


def test_interlock_outranks_disabled_message() -> None:
    # A live-port misconfiguration must be the loudest message, even while
    # execution is also disabled.
    result = can_execute(_settings(enabled=False, port=4001), NOT_KILLED)
    assert result.allowed is False
    assert "4001" in result.reason


def test_kill_outranks_disabled_message() -> None:
    result = can_execute(_settings(enabled=False), KILLED)
    assert result.allowed is False
    assert "kill" in result.reason.lower()


def test_paper_only_false_bypasses_interlock() -> None:
    # Deliberate, documented live escape hatch: requires BOTH paper_only=False
    # AND enabled=True — two explicit config flips. Out of scope for the paper
    # epic but the gate's behavior is pinned here.
    result = can_execute(
        _settings(enabled=True, paper_only=False, paper=False, port=4001),
        NOT_KILLED,
    )
    assert result.allowed is True
