"""Tests for the Settings class."""

from pathlib import Path

import pytest

from optionsbot.config import (
    IBKRSettings,
    ScanSettings,
    Settings,
    StorageSettings,
    TelegramSettings,
    load_settings,
)


def test_default_settings_are_valid() -> None:
    s = Settings()
    assert isinstance(s.ibkr, IBKRSettings)
    assert isinstance(s.telegram, TelegramSettings)
    assert isinstance(s.scan, ScanSettings)
    assert isinstance(s.storage, StorageSettings)


def test_manage_settings_defaults() -> None:
    s = Settings()
    assert s.manage.enabled is True
    assert s.manage.manage_dte == 21 and s.manage.urgent_dte == 7
    assert s.manage.assignment_alerts is True and s.manage.cooldown_hours == 24
    assert s.manage.profit_alerts is True
    assert s.manage.take_profit_pct == 0.5 and s.manage.stop_loss_mult == 2.0
    assert s.manage.min_credit == 20.0
    assert s.manage.debit_take_profit_pct == 0.5 and s.manage.debit_stop_pct == 0.5
    assert s.manage.min_debit == 20.0


def test_validation_settings_default() -> None:
    assert Settings().validation.outcomes_eval_hours == 24


def test_manage_long_leg_expiry_alerts_default_on() -> None:
    assert Settings().manage.long_leg_expiry_alerts is True


def test_manage_long_leg_expiry_alerts_overridable() -> None:
    from optionsbot.config import ManageSettings

    assert ManageSettings(long_leg_expiry_alerts=False).long_leg_expiry_alerts is False


def test_portfolio_settings_defaults() -> None:
    s = Settings()
    assert s.portfolio.enabled is True
    assert s.portfolio.beta_window == 252


def test_portfolio_beta_window_rejects_below_two() -> None:
    from pydantic import ValidationError

    from optionsbot.config import PortfolioSettings

    with pytest.raises(ValidationError):
        PortfolioSettings(beta_window=1)


def test_ibkr_paper_defaults_to_true() -> None:
    assert Settings().ibkr.paper is True


def test_ibkr_port_defaults_to_paper_4002() -> None:
    assert Settings().ibkr.port == 4002


def test_ibkr_max_market_data_lines_default_is_50() -> None:
    assert Settings().ibkr.max_market_data_lines == 50


def test_ibkr_max_market_data_lines_rejects_below_one() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IBKRSettings(max_market_data_lines=0)


def test_scan_threshold_default_is_55() -> None:
    # IBK-100: repurposed as the alert quality floor (was 70)
    assert Settings().scan.score_threshold == 55


def test_scan_interval_default_is_15_minutes() -> None:
    assert Settings().scan.interval_minutes == 15


def test_scan_auto_screen_default_is_on() -> None:
    # IBK-101: the daemon screens the universe each tick by default.
    assert Settings().scan.auto_screen is True


def test_scan_risk_pct_default_is_2_percent() -> None:
    assert Settings().scan.risk_pct == 0.02


def test_scan_risk_pct_rejects_out_of_range() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScanSettings(risk_pct=0.0)
    with pytest.raises(ValidationError):
        ScanSettings(risk_pct=1.5)


def test_scan_back_month_dte_gap_default_is_30() -> None:
    assert Settings().scan.back_month_dte_gap == 30


def test_scan_back_month_dte_gap_rejects_below_one() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScanSettings(back_month_dte_gap=0)


def test_env_var_overrides_nested_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONSBOT_IBKR__PORT", "7497")
    monkeypatch.setenv("OPTIONSBOT_IBKR__PAPER", "false")
    s = Settings()
    assert s.ibkr.port == 7497
    assert s.ibkr.paper is False


def test_storage_db_path_is_under_user_home() -> None:
    s = Settings()
    assert "optionsbot" in str(s.storage.db_path)
    # Should be an absolute path.
    assert Path(s.storage.db_path).is_absolute()


def test_load_settings_respects_explicit_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[ibkr]\n"
        "host = \"10.0.0.5\"\n"
        "port = 7496\n"
        "paper = false\n"
    )
    s = load_settings(config_file=cfg)
    assert s.ibkr.host == "10.0.0.5"
    assert s.ibkr.port == 7496
    assert s.ibkr.paper is False


def test_telegram_token_optional_in_defaults() -> None:
    # Defaults should not require a token; presence is enforced only when
    # the notify layer actually tries to send.
    s = Settings()
    assert s.telegram.bot_token is None
    assert s.telegram.chat_id is None


def test_telegram_heartbeat_minutes_default_is_60() -> None:
    assert Settings().telegram.heartbeat_minutes == 60


def test_telegram_heartbeat_minutes_rejects_negative() -> None:
    import pytest
    from pydantic import ValidationError

    from optionsbot.config import TelegramSettings

    with pytest.raises(ValidationError):
        TelegramSettings(heartbeat_minutes=-1)


def test_get_settings_cache_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    from optionsbot.config import get_settings

    get_settings.cache_clear()
    first = get_settings()
    monkeypatch.setenv("OPTIONSBOT_IBKR__PORT", "7497")

    # Without clearing, the cached value is returned even after the env change.
    assert get_settings() is first
    assert get_settings().ibkr.port == first.ibkr.port

    # After clearing, the new env is picked up.
    get_settings.cache_clear()
    refreshed = get_settings()
    assert refreshed.ibkr.port == 7497

    # Leave the cache empty so other tests start fresh.
    get_settings.cache_clear()


def test_load_settings_raises_helpful_error_for_malformed_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text("ibkr = [unterminated\n")  # malformed
    with pytest.raises(ValueError, match="Failed to parse TOML config"):
        load_settings(config_file=cfg)


def test_env_var_beats_toml_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[ibkr]\n"
        "port = 7496\n"  # TOML says 7496
    )
    monkeypatch.setenv("OPTIONSBOT_IBKR__PORT", "4001")  # env says 4001
    s = load_settings(config_file=cfg)
    assert s.ibkr.port == 4001  # env wins


def test_lowercase_env_var_still_beats_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Settings has case_sensitive=False by default, so lowercase env vars
    # must override TOML too. (Regression test for the precedence helper.)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[ibkr]\nport = 7496\n")
    monkeypatch.setenv("optionsbot_ibkr__port", "4001")
    s = load_settings(config_file=cfg)
    assert s.ibkr.port == 4001


def test_env_var_beats_toml_for_top_level_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Top-level (non-nested) fields like log_level should follow the same
    # env > TOML > defaults order.
    cfg = tmp_path / "config.toml"
    cfg.write_text('log_level = "WARNING"\n')
    monkeypatch.setenv("OPTIONSBOT_LOG_LEVEL", "DEBUG")
    s = load_settings(config_file=cfg)
    assert s.log_level == "DEBUG"


def test_documented_telegram_env_vars_actually_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: README and .env.example must document env var names that
    # the loader actually recognises. The pre-fix bug had bare
    # TELEGRAM_BOT_TOKEN in both files, which pydantic-settings silently
    # ignored due to the OPTIONSBOT_ prefix + __ nested delimiter on
    # Settings. This test pins the contract for the documented names.
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__BOT_TOKEN", "abc:secret123")
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__CHAT_ID", "-1009876")
    s = Settings()
    assert s.telegram.bot_token == "abc:secret123"
    assert s.telegram.chat_id == "-1009876"


def test_bare_telegram_env_vars_are_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Documents the (deliberate) behavior: bare TELEGRAM_BOT_TOKEN is NOT
    # picked up. The env_prefix on Settings enforces the namespaced form.
    # If we ever add validation_alias to also accept bare names, this
    # test will need updating to reflect the new contract.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "should-be-ignored")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "ignored-too")
    s = Settings()
    assert s.telegram.bot_token is None
    assert s.telegram.chat_id is None


def test_ibkr_client_id_exec_default_is_3() -> None:
    # IBK-125: dedicated execution clientId — order events are only visible
    # to the placing clientId, so the OrderClient never shares MCP=1/daemon=2.
    assert Settings().ibkr.client_id_exec == 3


def test_execution_settings_defaults() -> None:
    # IBK-123: execution is OFF by default — without explicit opt-in the bot
    # stays analysis/alerting-only, exactly as before the execution epic.
    s = Settings()
    assert s.execution.enabled is False
    assert s.execution.mode == "confirm"
    assert s.execution.paper_only is True
    assert s.execution.max_open_positions == 6
    assert s.execution.max_per_symbol == 1
    assert s.execution.max_daily_loss_pct == 0.02
    assert s.execution.max_consecutive_losses == 4


def test_execution_enabled_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONSBOT_EXECUTION__ENABLED", "true")
    assert Settings().execution.enabled is True


def test_execution_mode_rejects_unknown_value() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(mode="yolo")  # type: ignore[arg-type]


def test_execution_bounds_reject_out_of_range() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(max_daily_loss_pct=0.0)
    with pytest.raises(ValidationError):
        ExecutionSettings(max_daily_loss_pct=1.5)
    with pytest.raises(ValidationError):
        ExecutionSettings(max_open_positions=0)
    with pytest.raises(ValidationError):
        ExecutionSettings(max_consecutive_losses=0)
