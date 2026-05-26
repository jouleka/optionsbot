"""Configuration for optionsbot.

Resolution order (highest priority first):
  1. Environment variables (prefix OPTIONSBOT_, nested via __).
  2. Values in ~/.config/optionsbot/config.toml.
  3. Defaults defined on the Settings classes below.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_FILE = Path.home() / ".config" / "optionsbot" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "optionsbot" / "optionsbot.db"


class IBKRSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4002  # IB Gateway paper default (see IBK-16 for port conventions)
    client_id_mcp: int = 1
    client_id_daemon: int = 2
    paper: bool = True


class TelegramSettings(BaseModel):
    bot_token: str | None = None
    chat_id: str | None = None


class ScanSettings(BaseModel):
    interval_minutes: int = Field(default=15, ge=1)
    score_threshold: int = Field(default=70, ge=0, le=100)
    alert_cooldown_hours: int = Field(default=4, ge=0)
    alert_rescore_delta: int = Field(default=10, ge=0, le=100)


class StorageSettings(BaseModel):
    db_path: Path = DEFAULT_DB_PATH


class Settings(BaseSettings):
    """Top-level settings loaded from env + optional config.toml."""

    ibkr: IBKRSettings = IBKRSettings()
    telegram: TelegramSettings = TelegramSettings()
    scan: ScanSettings = ScanSettings()
    storage: StorageSettings = StorageSettings()

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="OPTIONSBOT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_settings(config_file: Path | None = None) -> Settings:
    """Load Settings with optional config.toml overlay.

    Resolution: TOML values fill defaults; env vars (via Settings()) then override TOML.
    """
    cfg_path = config_file if config_file is not None else DEFAULT_CONFIG_FILE
    toml_data = _load_toml(cfg_path)
    if toml_data:
        return Settings(**toml_data)
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton for use across the package."""
    return load_settings()
