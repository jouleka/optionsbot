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

from pydantic import BaseModel, Field, model_validator
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
    # IBK-122: IBKR reqMarketDataType code — 1=live, 2=frozen, 3=delayed,
    # 4=delayed-frozen. Applied in paper mode (see ibkr/client.py). Default 3
    # (delayed) is the safe pre-IBK-122 behavior; flip to 1 once live data is
    # shared into the paper account.
    market_data_type: Literal[1, 2, 3, 4] = 3


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
    # IBK-149 (scan-resilience): bound blocking external calls so a flaky
    # dependency degrades the scan gracefully instead of stalling it (and the
    # asyncio loop). scan_symbol_timeout_s caps any per-symbol awaitable hang
    # (e.g. a wedged IB Gateway); screen_timeout_s caps the universe screen
    # (falls back to watchlist-only); external_data_timeout_s bounds the
    # off-loop yfinance earnings/news calls.
    scan_symbol_timeout_s: float = Field(default=30.0, gt=0.0)
    screen_timeout_s: float = Field(default=60.0, gt=0.0)
    external_data_timeout_s: float = Field(default=5.0, gt=0.0)


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
    # Liquidity gates — proportional, the way a trader actually judges them.
    # Per-leg is a lenient SANITY check (catch broken/garbage quotes): a leg's
    # bid/ask spread must be within max(frac x leg mid, floor $), so a pricey
    # but liquid leg isn't killed by an absolute dollar cap. The real ECONOMIC
    # gate is combo-level: the combo's bid/ask spread must be <= combo_frac of
    # the net premium, so slippage can't eat the credit (and we can exit). The
    # price-walk remains the hard backstop against overpaying on entry.
    max_leg_spread_frac: float = Field(default=0.40, gt=0.0)
    max_leg_spread_floor: float = Field(default=0.20, gt=0.0)
    max_combo_spread_frac: float = Field(default=0.35, gt=0.0)
    # min_open_interest=0 disables the OI check — delayed snapshots often omit
    # OI; enable once real-time data is shared to paper (IBK-122).
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
    # IBK-PHASE0-C1: quote-freshness gate for QUOTE-PRICED exits (TP/soft
    # stop). If the option snapshots driving an exit decision are older than
    # this many seconds, the quote-priced exit is SUPPRESSED for that tick
    # (never price a close off a stale mid / cross a moved spread). The
    # time-based expiry/DTE guard is unaffected and always fires. 0 disables
    # the freshness gate (legacy behavior). Sized to the exit cadence and the
    # delayed-feed latency.
    exit_quote_max_age_seconds: int = Field(default=45, ge=0)
    # IBK-130 full-auto gates: reject new auto entries once this fraction of
    # net liquidation is already deployed ((net_liq - available)/net_liq).
    # Confirm-mode /execute is NOT bound by it — the human decides.
    max_bp_usage_pct: float = Field(default=0.30, gt=0.0, le=1.0)
    # IBK-133 dynamic sizing (authoritative at execution; scan.risk_pct only
    # shapes the alert's indicative size): base risk × quarter-Kelly edge
    # tilt × anti-martingale drawdown governor, capped by portfolio heat and
    # a single-trade ceiling. Min 1 lot when within both caps.
    base_risk_pct: float = Field(default=0.03, gt=0.0, le=1.0)
    max_portfolio_heat_pct: float = Field(default=0.15, gt=0.0, le=1.0)
    max_single_trade_risk_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    # PHASE 0 net-liq circuit breaker (work-stream B). The equity guard trips
    # the kill switch on a realized+unrealized day-start drawdown >=
    # max_daily_loss_pct, and blocks NEW entries once the drawdown reaches
    # entry_block_loss_frac of that cap (so the bot stops adding risk while it
    # is already bleeding, before the hard kill).
    entry_block_loss_frac: float = Field(default=0.75, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_phase0_ceilings(self) -> ExecutionSettings:
        # Phase 0 hard ceilings: reject a config that lifts the risk caps
        # past what is safe to run unattended 24/7. These are absolute
        # upper bounds — the per-field Field(le=...) bounds are looser
        # ranges; this guard is the safety backstop, applied on load.
        ceilings: tuple[tuple[str, float], ...] = (
            ("base_risk_pct", 0.05),
            # Aggregate defined-risk exposure may use at most half of live
            # USD net liquidation. The separate 15% single-trade ceiling keeps
            # one position from consuming the entire account-relative budget.
            ("max_portfolio_heat_pct", 0.50),
            ("max_single_trade_risk_pct", 0.15),
            ("max_bp_usage_pct", 0.50),
        )
        for name, ceiling in ceilings:
            value = getattr(self, name)
            if value > ceiling:
                raise ValueError(
                    f"execution.{name}={value} exceeds the Phase 0 safety "
                    f"ceiling of {ceiling}"
                )
        return self


class MonitorSettings(BaseModel):
    # IBK-137 Increment 2: gateway health paging. The daemon pages the human via
    # Telegram when the Gateway is WEDGED (a majority of scanned symbols dying on
    # the IBK-149 per-symbol budget during RTH -- connected but option data dead)
    # or DISCONNECTED during RTH with open positions (exit protection is DOWN).
    enabled: bool = True
    # Minimum per-symbol budget timeouts in one scan before a wedge can page
    # (floor so a tiny scan can't trip it); a MAJORITY of symbols must also fail.
    wedge_min_budget_timeouts: int = Field(default=3, ge=1)
    # Re-page cadence while a condition persists (first page is immediate).
    page_repeat_minutes: int = Field(default=30, ge=1)


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
    monitor: MonitorSettings = MonitorSettings()

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
