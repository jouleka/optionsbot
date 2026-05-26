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


def test_scan_threshold_default_is_70() -> None:
    assert Settings().scan.score_threshold == 70


def test_scan_interval_default_is_15_minutes() -> None:
    assert Settings().scan.interval_minutes == 15


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
    assert s.telegram.bot_token in (None, "")
    assert s.telegram.chat_id in (None, "")
