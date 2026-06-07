"""Configuration for optionsbot.

Resolution order (highest priority first):
  1. Environment variables (prefix OPTIONSBOT_, nested via __).
  2. Values in ~/.config/optionsbot/config.toml (or a path passed to load_settings).
  3. Defaults defined on the Settings classes below.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = Path.home() / ".config" / "optionsbot" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "optionsbot" / "optionsbot.db"


class IBKRSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4002  # IB Gateway paper default (see IBK-16 for port conventions)
    client_id_mcp: int = 1
    client_id_daemon: int = 2
    paper: bool = True
    # Max simultaneous streaming market-data lines to subscribe at once when
    # fetching an option chain. IBKR's default account allowance is 100; a
    # conservative 50 stays safely under it. Bump if your data tier permits.
    max_market_data_lines: int = Field(default=50, ge=1)


class TelegramSettings(BaseModel):
    bot_token: str | None = None
    chat_id: str | None = None
    # IBK-102: post a periodic "alive + last tick" summary to Telegram during
    # market hours. 0 disables.
    heartbeat_minutes: int = Field(default=60, ge=0)


class ScanSettings(BaseModel):
    interval_minutes: int = Field(default=15, ge=1)
    # Relative strength (IBK-109): each symbol's return minus this benchmark's over
    # relative_strength_window trading days, surfaced in daily_brief as context.
    benchmark_symbol: str = Field(default="SPY")
    relative_strength_window: int = Field(default=20, ge=2)
    # Daemon alert quality FLOOR (IBK-100): a pick must score >= this to be
    # alert-worthy. Repurposed from the old absolute threshold (was 70). scan-once
    # and the MCP analyze path use scoring.DEFAULT_THRESHOLD, not this field.
    score_threshold: int = Field(default=55, ge=0, le=100)
    # Daemon alerts the top-N highest-scored floor-passing picks per tick, ranked
    # across all scanned symbols (not per-symbol).
    alert_top_n: int = Field(default=3, ge=1)
    alert_cooldown_hours: int = Field(default=4, ge=0)  # 0 disables the cooldown
    alert_rescore_delta: int = Field(default=10, ge=0, le=100)
    # Chain fetch: bound the strike set to a near-ATM window so a scan doesn't
    # pull the full ladder (~497 strikes for SPY) x every in-window expiry.
    strike_band_pct: float = Field(default=0.15, gt=0.0, le=1.0)
    max_strikes_per_side: int = Field(default=40, ge=1)
    # Per scan, fetch a near-target front expiry + the nearest back-month at
    # least this many DTE beyond it (so Calendar/Diagonal spreads stay viable;
    # matches strategies.calendar._MIN_BACK_OVER_FRONT_DTE).
    back_month_dte_gap: int = Field(default=30, ge=1)
    # Per-trade position sizing: fraction of account net-liquidation used as the
    # risk budget (budget = net_liq * risk_pct; contracts = budget // max_loss).
    risk_pct: float = Field(default=0.02, gt=0.0, le=1.0)
    # Auto-screen the universe each tick and scan the top screened candidates
    # in addition to the watchlist (IBK-101). False = watchlist-only tick (the
    # pre-IBK-101 behavior).
    auto_screen: bool = True


class StorageSettings(BaseModel):
    db_path: Path = DEFAULT_DB_PATH


class ScreenerSettings(BaseModel):
    # When set, REPLACES screener.universe.DEFAULT_UNIVERSE.
    universe: list[str] | None = None
    # Stage-1 liquidity gate: minimum trailing average daily dollar volume.
    min_dollar_volume: float = Field(default=5_000_000.0, ge=0.0)
    # Default number of candidates the `screen` command prints.
    top_n: int = Field(default=20, ge=1)
    # Stage-2: how many of the top screened candidates to full-scan with
    # `screen --scan`. Each is a full option-chain fetch (~40-50s), so keep it
    # small for pacing. Overridable per-run with --scan-top.
    scan_top_n: int = Field(default=5, ge=1)


class Settings(BaseSettings):
    """Top-level settings.

    Source priority (highest first): init kwargs, env vars (OPTIONSBOT_FOO__BAR),
    .env file, then the optional TOML config file. Defaults are the lowest.
    """

    ibkr: IBKRSettings = IBKRSettings()
    telegram: TelegramSettings = TelegramSettings()
    scan: ScanSettings = ScanSettings()
    screener: ScreenerSettings = ScreenerSettings()
    storage: StorageSettings = StorageSettings()

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="OPTIONSBOT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        toml_file=None,  # populated dynamically below
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_path = cls.model_config.get("toml_file")
        sources: tuple[PydanticBaseSettingsSource, ...] = (
            init_settings,
            env_settings,
            dotenv_settings,
        )
        if isinstance(toml_path, (str, Path)):
            # Validate up front so we get our friendlier error message rather than
            # whatever pydantic-settings produces when its lazy parse fails.
            _validate_toml_file(Path(toml_path))
            sources = sources + (TomlConfigSettingsSource(settings_cls, toml_file=toml_path),)
        sources = sources + (file_secret_settings,)
        return sources


def _validate_toml_file(path: Path) -> None:
    """Raise a friendly ValueError if the TOML file is malformed."""
    if not path.exists():
        return
    with path.open("rb") as f:
        try:
            tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Failed to parse TOML config at {path}: {e}") from e


def load_settings(config_file: Path | None = None) -> Settings:
    """Load Settings with an optional TOML overlay.

    Resolution: env > TOML > defaults. Pass ``config_file`` to point at a
    specific TOML file; pass ``None`` (default) to use
    ``~/.config/optionsbot/config.toml`` if it exists.
    """
    cfg_path = config_file if config_file is not None else DEFAULT_CONFIG_FILE
    # Only attach the TOML source when the file actually exists -- otherwise
    # pydantic-settings emits a noisy "file not found" log for the common
    # "no config.toml yet" case.
    if cfg_path.exists():
        # Mutate model_config for THIS instantiation only. We restore after.
        previous = Settings.model_config.get("toml_file")
        Settings.model_config["toml_file"] = str(cfg_path)
        try:
            return Settings()
        finally:
            Settings.model_config["toml_file"] = previous
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton for use across the package.

    The cache is populated on the first call and not invalidated when
    environment variables change at runtime. Tests or any code that
    mutates the environment between calls must invoke
    ``get_settings.cache_clear()`` to force a re-read on the next call.

    For long-running processes (e.g., the daemon in IBK-7) that should
    pick up edits to ``~/.config/optionsbot/config.toml`` without a
    restart, register a SIGHUP handler that clears the cache::

        import signal
        from optionsbot.config import get_settings

        def _on_sighup(signum, frame):
            get_settings.cache_clear()

        signal.signal(signal.SIGHUP, _on_sighup)
    """
    return load_settings()
