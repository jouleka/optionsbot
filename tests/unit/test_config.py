"""Tests for the Settings class."""

import math
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
    assert s.hermes_webhook.enabled is False
    assert isinstance(s.scan, ScanSettings)
    assert isinstance(s.storage, StorageSettings)
    assert s.execution.zero_dte_debit_max_profit_take_pct == 0.50
    assert s.execution.zero_dte_debit_trail_early_giveback_pct == 0.35
    assert s.execution.zero_dte_debit_trail_late_giveback_pct == 0.10


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


def test_hermes_entry_review_bypass_requires_paper_only() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        from optionsbot.config import ExecutionSettings

        ExecutionSettings(
            paper_only=False,
            require_hermes_entry_review=False,
        )


def test_auto_earnings_opt_in_requires_paper_only() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        from optionsbot.config import ExecutionSettings

        ExecutionSettings(
            paper_only=False,
            auto_skip_earnings=False,
        )


def test_structural_margin_fallback_requires_paper_only() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        from optionsbot.config import ExecutionSettings

        ExecutionSettings(
            paper_only=False,
            allow_structural_margin_fallback=True,
        )


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


def test_scan_dte_defaults() -> None:
    scan = Settings().scan
    assert scan.dte_target == 45
    assert scan.dte_window_min == 25
    assert scan.dte_window_max == 55


def test_scan_dte_window_accepts_aggressive_profile() -> None:
    scan = ScanSettings(dte_target=21, dte_window_min=10, dte_window_max=35)
    assert (scan.dte_window_min, scan.dte_target, scan.dte_window_max) == (10, 21, 35)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dte_window_min": 22, "dte_target": 21, "dte_window_max": 35},
        {"dte_window_min": 10, "dte_target": 36, "dte_window_max": 35},
    ],
)
def test_scan_dte_window_rejects_target_outside_window(kwargs: dict[str, int]) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScanSettings(**kwargs)


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


def test_execution_semi_auto_defaults() -> None:
    # IBK-126
    s = Settings()
    assert s.execution.max_pick_age_minutes == 20
    assert s.execution.order_ttl_minutes == 10
    assert s.execution.credit_drift_warn_pct == 0.25


def test_execution_reconcile_default() -> None:
    # IBK-128
    assert Settings().execution.reconcile_minutes == 5


def test_execution_auto_defaults() -> None:
    # IBK-130
    assert Settings().execution.max_bp_usage_pct == 0.30


def test_execution_sizing_defaults() -> None:
    # IBK-133
    s = Settings()
    assert s.execution.base_risk_pct == 0.03
    assert s.execution.max_portfolio_heat_pct == 0.15
    assert s.execution.max_single_trade_risk_pct == 0.10


def test_execution_exit_defaults() -> None:
    # IBK-129: soft stop OFF by default (defined-risk width is the stop);
    # expiry guard force-closes 3 days out.
    s = Settings()
    assert s.execution.exit_stop_enabled is False
    assert s.execution.expiry_guard_dte == 3


def test_execution_walk_defaults() -> None:
    # IBK-127
    s = Settings()
    assert s.execution.walk_step_seconds == 10
    assert s.execution.walk_max_steps == 4
    assert s.execution.walk_final_rest_seconds == 120
    assert s.execution.max_slippage_spread_frac == 0.25
    assert s.execution.max_slippage_abs == 0.10
    assert s.execution.max_leg_spread_frac == 0.40
    assert s.execution.max_leg_spread_floor == 0.20
    assert s.execution.max_combo_spread_frac == 0.35
    assert s.execution.min_open_interest == 0


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


def test_execution_rejects_base_risk_above_ceiling() -> None:
    # Phase 0 safety: base_risk_pct must not exceed 0.05.
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(base_risk_pct=0.25)


def test_execution_rejects_heat_above_ceiling() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(max_portfolio_heat_pct=1.0)


def test_execution_rejects_single_trade_above_ceiling() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(max_single_trade_risk_pct=math.nextafter(0.10, math.inf))


def test_execution_rejects_operational_risk_caps_above_hard_ceilings() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    unsafe = (
        {"max_daily_loss_pct": 0.0500001},
        {"max_open_positions": 11},
        {"max_per_symbol": 4},
        {"max_consecutive_losses": 11},
    )
    for override in unsafe:
        with pytest.raises(ValidationError):
            ExecutionSettings(**override)  # type: ignore[arg-type]


def test_execution_rejects_bp_usage_above_ceiling() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(max_bp_usage_pct=0.95)


def test_execution_rejects_freshness_values_above_hard_ceilings() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    unsafe = (
        {"max_pick_age_minutes": 61},
        {"entry_quote_max_age_seconds": 121},
        {"exit_quote_max_age_seconds": 121},
    )
    for override in unsafe:
        with pytest.raises(ValidationError):
            ExecutionSettings(**override)  # type: ignore[arg-type]


def test_execution_rejects_disabled_exit_quote_freshness() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(exit_quote_max_age_seconds=0)


def test_freshness_bounds_apply_to_toml(tmp_path: Path) -> None:
    from pydantic import ValidationError

    cfg = tmp_path / "config.toml"
    cfg.write_text("[execution]\nexit_quote_max_age_seconds = 0\n")

    with pytest.raises(ValidationError):
        load_settings(config_file=cfg)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPTIONSBOT_EXECUTION__MAX_PICK_AGE_MINUTES", "61"),
        ("OPTIONSBOT_EXECUTION__ENTRY_QUOTE_MAX_AGE_SECONDS", "121"),
        ("OPTIONSBOT_EXECUTION__EXIT_QUOTE_MAX_AGE_SECONDS", "121"),
    ],
)
def test_freshness_bounds_apply_to_env(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings()


def test_execution_accepts_values_at_the_ceiling() -> None:
    # The ceilings themselves are valid (<=, not <). Portfolio heat may use
    # up to half of live USD net liquidation, but no single trade gets that cap.
    from optionsbot.config import ExecutionSettings

    e = ExecutionSettings(
        base_risk_pct=0.05,
        max_portfolio_heat_pct=0.50,
        max_single_trade_risk_pct=0.10,
        max_bp_usage_pct=0.50,
        max_pick_age_minutes=60,
        entry_quote_max_age_seconds=120,
        exit_quote_max_age_seconds=120,
    )
    assert e.base_risk_pct == 0.05
    assert e.max_portfolio_heat_pct == 0.50
    assert e.max_single_trade_risk_pct == 0.10
    assert e.max_bp_usage_pct == 0.50
    assert e.max_pick_age_minutes == 60
    assert e.entry_quote_max_age_seconds == 120
    assert e.exit_quote_max_age_seconds == 120


def test_execution_rejects_portfolio_heat_above_half_account() -> None:
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(max_portfolio_heat_pct=0.500001)


def test_aggressive_config_file_fails_to_load(tmp_path: Path) -> None:
    # A config that lifts the caps the way the old aggressive live file did
    # must be REJECTED at load time, not silently accepted.
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[execution]\n"
        "enabled = true\n"
        "base_risk_pct = 0.25\n"
        "max_single_trade_risk_pct = 1.0\n"
        "max_portfolio_heat_pct = 1.0\n"
        "max_bp_usage_pct = 0.95\n"
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_settings(config_file=cfg)


def test_entry_block_loss_frac_defaults_to_three_quarters() -> None:
    from optionsbot.config import Settings

    s = Settings()
    assert s.execution.entry_block_loss_frac == 0.75


def test_entry_block_loss_frac_is_a_fraction() -> None:
    import pytest
    from pydantic import ValidationError

    from optionsbot.config import ExecutionSettings

    with pytest.raises(ValidationError):
        ExecutionSettings(entry_block_loss_frac=0.0)
    with pytest.raises(ValidationError):
        ExecutionSettings(entry_block_loss_frac=1.5)


def test_ibkr_market_data_type_default_is_delayed() -> None:
    from optionsbot.config import IBKRSettings

    assert IBKRSettings().market_data_type == 3


def test_ibkr_market_data_type_accepts_live() -> None:
    from optionsbot.config import IBKRSettings

    assert IBKRSettings(market_data_type=1).market_data_type == 1


def test_ibkr_market_data_type_rejects_out_of_range() -> None:
    from pydantic import ValidationError

    from optionsbot.config import IBKRSettings

    with pytest.raises(ValidationError):
        IBKRSettings(market_data_type=5)
