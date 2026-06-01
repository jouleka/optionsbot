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


def test_scan_threshold_default_is_70() -> None:
    assert Settings().scan.score_threshold == 70


def test_scan_interval_default_is_15_minutes() -> None:
    assert Settings().scan.interval_minutes == 15


def test_scan_risk_pct_default_is_2_percent() -> None:
    assert Settings().scan.risk_pct == 0.02


def test_scan_risk_pct_rejects_out_of_range() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScanSettings(risk_pct=0.0)
    with pytest.raises(ValidationError):
        ScanSettings(risk_pct=1.5)


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
