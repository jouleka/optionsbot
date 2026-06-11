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
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = Path.home() / ".config" / "optionsbot" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "optionsbot" / "optionsbot.db"

# Recognized paper ports: IB Gateway paper / TWS paper (IBK-16 conventions).
# Lives here (zero-dependency module) so both the execution gate and the
# ibkr-layer order interlock can share it without an import cycle.
PAPER_PORTS: frozenset[int] = frozenset({4002, 7497})


class IBKRSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4002  # IB Gateway paper default (see IBK-16 for port conventions)
    client_id_mcp: int = 1
    client_id_daemon: int = 2
    # IBK-125: dedicated execution clientId — order status/fill events are
    # only delivered to the clientId that placed the order.
    client_id_exec: int = 3
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


class ManageSettings(BaseModel):
    # IBK-113: proactive position-management alerts. enabled=False disables the whole
    # management pass; thresholds are calendar DTE; cooldown_hours bounds re-alerts for
    # the same (leg, trigger) -- persisted in position_alerts so it survives restarts.
    enabled: bool = True
    manage_dte: int = Field(default=21, ge=0)
    urgent_dte: int = Field(default=7, ge=0)
    assignment_alerts: bool = True
    # IBK-119: also alert on LONG option legs approaching expiry (DTE buckets, ITM-aware
    # wording). Assignment stays short-only. False = short-leg-only management (pre-IBK-119).
    long_leg_expiry_alerts: bool = True
    cooldown_hours: int = Field(default=24, ge=0)
    # IBK-114: per-underlying take-profit / stop-loss on net-credit positions.
    profit_alerts: bool = True
    take_profit_pct: float = Field(default=0.5, gt=0.0)  # alert at >= this fraction of credit
    stop_loss_mult: float = Field(default=2.0, gt=0.0)  # alert at <= -this multiple of credit
    # Skip groups below this net credit ($). Non-zero by default so a near-zero balanced /
    # rolled book (net credit ~ a few cents) can't produce nonsense "50000% of $0" alerts.
    min_credit: float = Field(default=20.0, ge=0.0)
    # IBK-116: net-DEBIT positions (long calls/puts, debit spreads) -- % return on debit paid.
    debit_take_profit_pct: float = Field(default=0.5, gt=0.0)  # +X% on the debit paid
    debit_stop_pct: float = Field(default=0.5, gt=0.0)  # -X% of the debit paid
    min_debit: float = Field(default=20.0, ge=0.0)  # skip groups below this net debit ($)


class ValidationSettings(BaseModel):
    # IBK-117: how often the daemon evaluates newly-expired picks into the outcomes ledger.
    outcomes_eval_hours: int = Field(default=24, ge=0)  # 0 disables the daily accrual job


class PortfolioSettings(BaseModel):
    # IBK-118: beta-weighted portfolio delta on the open book. enabled=False skips the
    # extra per-underlying history/spot I/O (no `beta_weighted` on the view). The benchmark
    # reuses scan.benchmark_symbol (one benchmark concept across the bot).
    enabled: bool = True
    beta_window: int = Field(default=252, ge=2)  # trading days of daily returns for beta


class ExecutionSettings(BaseModel):
    # IBK-123: automated order execution (paper-first). enabled=False means the
    # bot NEVER places orders regardless of any other setting — exactly the
    # pre-execution-epic behavior (analysis + alerting only).
    enabled: bool = False
    # confirm = orders happen only via an explicit Telegram /execute on an
    # alerted pick (IBK-126); auto = the scan tick may stage entries itself
    # (IBK-130). Declared now so config keys stay stable across the epic.
    mode: Literal["confirm", "auto"] = "confirm"
    # Hard interlock: while True, can_execute refuses to arm unless
    # ibkr.paper=True AND ibkr.port is a recognized paper port (4002 Gateway /
    # 7497 TWS). Flipping this off is a deliberate live-trading decision and
    # out of scope for the paper epic.
    paper_only: bool = True
    # Portfolio caps consumed by the entry gates (IBK-126/130).
    max_open_positions: int = Field(default=6, ge=1)
    max_per_symbol: int = Field(default=1, ge=1)
    # Kill-switch thresholds: auto-trip wiring lands in IBK-130; the persisted
    # switch + Telegram /kill work from IBK-123.
    max_daily_loss_pct: float = Field(default=0.02, gt=0.0, le=1.0)
    max_consecutive_losses: int = Field(default=4, ge=1)
    # IBK-126 semi-auto execution. A pick older than max_pick_age is the wrong
    # trade (strikes/credit moved) — rescan instead. v1 pricing places at the
    # fresh mid and rests until the TTL, then auto-cancels (= trade skipped);
    # the reprice ladder arrives in IBK-127. credit_drift_warn_pct flags when
    # the fresh mid drifted from the alerted credit.
    max_pick_age_minutes: int = Field(default=20, ge=1)
    order_ttl_minutes: int = Field(default=10, ge=1)
    credit_drift_warn_pct: float = Field(default=0.25, gt=0.0)
    # IBK-127 price walk: submit at mid, reprice toward marketable every
    # walk_step_seconds for walk_max_steps (0 disables walking), final price
    # rests walk_final_rest_seconds, then cancel = trade skipped. The hard
    # slippage budget is min(frac × combo spread, abs cap), at least one
    # price increment, anchored to the DECISION mid.
    walk_step_seconds: int = Field(default=10, ge=0)
    walk_max_steps: int = Field(default=4, ge=0)
    walk_final_rest_seconds: int = Field(default=120, ge=0)
    max_slippage_spread_frac: float = Field(default=0.25, gt=0.0, le=1.0)
    max_slippage_abs: float = Field(default=0.10, gt=0.0)
    # IBK-127 liquidity gates. min_open_interest=0 disables the OI check —
    # delayed snapshots often omit OI; enable once real-time data is shared
    # to paper (IBK-122).
    max_leg_spread_dollars: float = Field(default=0.50, gt=0.0)
    min_open_interest: int = Field(default=0, ge=0)
    # IBK-128: periodic broker reconciliation cadence (startup always runs
    # one pass). 0 disables the periodic pass; it only fires while
    # non-terminal orders exist, so it is free when idle.
    reconcile_minutes: int = Field(default=5, ge=0)
    # IBK-129 exits: soft stop OFF by default — the defined-risk width IS the
    # stop (gaps blow through soft stops anyway, per the research); profit
    # targets/DTE come from ManageSettings. The expiry guard force-closes any
    # bot position this many days before its NEAREST leg expiry.
    exit_stop_enabled: bool = False
    expiry_guard_dte: int = Field(default=3, ge=0)
    # IBK-130 full-auto gates: reject new auto entries once this fraction of
    # net liquidation is already deployed ((net_liq - available)/net_liq).
    # Confirm-mode /execute is NOT bound by it — the human decides.
    max_bp_usage_pct: float = Field(default=0.30, gt=0.0, le=1.0)


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
    manage: ManageSettings = ManageSettings()
    validation: ValidationSettings = ValidationSettings()
    portfolio: PortfolioSettings = PortfolioSettings()
    execution: ExecutionSettings = ExecutionSettings()

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
