# optionsbot

Personal IBKR options-analysis + alerting tool, with an opt-in
paper-only automated-execution mode (in progress, IBK-122..131).

## ⚠ Safety disclaimer

**This is not financial advice.** optionsbot is a personal analysis +
alerting tool with an opt-in automated-execution mode for IBKR PAPER
accounts. Order placement is OFF by default (`execution.enabled=false`).
When enabled, a hard interlock (`execution.paper_only`, default true)
refuses to place orders unless the configured connection is a paper
account on a recognized paper port (4002 Gateway / 7497 TWS), and a
persisted kill switch (Telegram `/kill`) halts execution at any time
and survives restarts.

- Paper trading only. Do not point this at a live IBKR account; the
  paper-only interlock exists precisely to refuse that.
- All alerts are informational. Past performance is not predictive.
- The scoring engine is a heuristic. It can be wrong. Verify every
  suggestion against your own analysis before acting on it.
- Automated paper trades are an experiment to MEASURE the strategy's
  edge, not proof of one. Paper fills are optimistic vs live.
- The author accepts no liability for trading losses incurred from
  using this tool. Read the source before relying on it.

## What it does

- Watches a configurable list of equity symbols.
- Every ~15 minutes during market hours, scans each ticker: fetches
  the option chain, computes IV rank / HV / market view, scores 16
  options strategies against the snapshot, and persists results to
  SQLite.
- Sends Telegram alerts when a strategy crosses the configured
  composite score threshold (default 70), with cooldown + score-delta
  deduplication so you don't get spammed.
- Exposes the watchlist + analysis as MCP tools so Claude Code can
  query the data interactively ("show me the top picks for AAPL").

Order execution is an opt-in epic in progress (IBK-122..131): with
`execution.enabled=true` the bot may place orders on the IBKR PAPER
account only — confirm-first via Telegram `/execute` before any full-auto
mode — behind the paper-only interlock and the `/kill` switch. As of
IBK-123 only the safety substrate exists; there is no order-placement
code yet.

## Architecture

Twelve in-tree code units, each with a single responsibility (a few are
modules rather than packages, hence "units" rather than strictly
"packages"):

| Package | Responsibility |
|---|---|
| `optionsbot.config` | pydantic-settings: env > TOML > defaults |
| `optionsbot.storage` | SQLAlchemy schema + alembic migrations |
| `optionsbot.ibkr` | adapter dataclasses + clients for `ib_async` |
| `optionsbot.analysis` | pure-function HV, IV rank, view inference |
| `optionsbot.strategies` | 16 strategies w/ ABC + factor weights |
| `optionsbot.scoring` | composite score + top-K selector + rationale |
| `optionsbot.scan` | end-to-end single-symbol scan helper |
| `optionsbot.alerts` | markdown alert formatter |
| `optionsbot.mcp_server` | FastMCP stdio server (7 tools for Claude) |
| `optionsbot.daemon` | APScheduler-driven scan loop + Telegram dispatch |
| `optionsbot.execution` | arming gate + persisted kill switch (no order code yet) |
| `optionsbot.observability` | structlog configuration + contextvars |

Async at the top (IBKR + Telegram + APScheduler are async). Analysis,
strategies, and scoring are sync. `optionsbot.ibkr` types are contained
to the IBKR layer; everything downstream uses the adapter dataclasses
(`StockQuote`, `OptionChainLeg`, `PositionRecord`, etc.) so a future
broker swap touches one package.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- IB Gateway (or TWS) on a paper account, reachable on the configured port
- (Optional) Telegram bot for alerts (see [docs/telegram-setup.md](docs/telegram-setup.md))

## Install

```bash
git clone git@github.com:jouleka/optionsbot.git
cd optionsbot
uv sync --group dev
```

## First-time setup

```bash
# Interactive wizard: writes ~/.config/optionsbot/config.toml,
# runs DB migrations, optionally tests Telegram.
uv run optionsbot init
```

The wizard is idempotent -- safe to re-run. Use `--non-interactive`
and `--skip-telegram` flags for CI / scripted setup.

For Telegram setup details (BotFather, chat_id), see
[docs/telegram-setup.md](docs/telegram-setup.md).

## Daily usage

### Health check

```bash
uv run optionsbot status         # pretty text, exit 0 if all critical subsystems ok
uv run optionsbot status --json  # machine-readable for monitoring pipelines
```

Reports: DB reachability, IB Gateway socket, last scan timestamp,
last alert timestamp, Telegram bot reachable.

### Run the daemon (foreground)

```bash
uv run optionsbot-daemon
```

For production -- auto-restart, journald logs, etc. -- run under
systemd-user. See [docs/systemd.md](docs/systemd.md).

### Claude Code integration

The MCP server exposes 7 tools (4 watchlist + analyze + latest_snapshot
+ score_breakdown) so Claude can query the data interactively. See
[docs/mcp-claude-code.md](docs/mcp-claude-code.md) for the
`mcpServers` config snippet.

### Manual watchlist management

The watchlist tools live in the MCP server; ask Claude to add/remove
tickers. The CLI `optionsbot watch add/remove/list` commands are
unimplemented stubs in v1 -- file an issue if you want them as a
non-Claude path.

## Configuration

Settings live in `~/.config/optionsbot/config.toml` and are overridable
by env vars with the prefix `OPTIONSBOT_` and nested-section delimiter
`__`. Examples:

| Setting | Env var | Default |
|---|---|---|
| IB Gateway host | `OPTIONSBOT_IBKR__HOST` | `127.0.0.1` |
| IB Gateway port | `OPTIONSBOT_IBKR__PORT` | `4002` (paper) |
| Scan interval | `OPTIONSBOT_SCAN__INTERVAL_MINUTES` | `15` |
| Score threshold | `OPTIONSBOT_SCAN__SCORE_THRESHOLD` | `70` |
| Alert cooldown | `OPTIONSBOT_SCAN__ALERT_COOLDOWN_HOURS` | `4` |
| Telegram token | `OPTIONSBOT_TELEGRAM__BOT_TOKEN` | _(unset)_ |
| Telegram chat | `OPTIONSBOT_TELEGRAM__CHAT_ID` | _(unset)_ |
| DB path | `OPTIONSBOT_STORAGE__DB_PATH` | `~/.local/share/optionsbot/optionsbot.db` |
| Log level | `OPTIONSBOT_LOG_LEVEL` | `INFO` |
| Execution master switch | `OPTIONSBOT_EXECUTION__ENABLED` | `false` |
| Execution mode | `OPTIONSBOT_EXECUTION__MODE` | `confirm` |
| Paper-only interlock | `OPTIONSBOT_EXECUTION__PAPER_ONLY` | `true` |

**Note the double underscore** between section and field. Bare names
like `TELEGRAM_BOT_TOKEN` are silently ignored.

IB Gateway port conventions (override `settings.ibkr.port` in config.toml if your install differs):

| App | Paper | Live |
|---|---|---|
| IB Gateway | 4002 | 4001 |
| TWS | 7497 | 7496 |

Distinct `client_id` values are reserved per process role so the MCP server and
the daemon can hold simultaneous connections without colliding:
`settings.ibkr.client_id_mcp` (default 1) and `settings.ibkr.client_id_daemon`
(default 2).

IBKR credentials are deliberately NOT in this repo. IB Gateway holds them.

## Architectural rules

These hold across every epic, including any future contributions:

1. **Execution is opt-in, paper-only, and gated.** `execution.enabled`
   defaults to false; every order-placing path must pass
   `optionsbot.execution.gate.can_execute` (paper-only interlock +
   persisted kill switch) before touching the IBKR order API. The
   pre-IBK-123 rule here was "no order placement, ever" — the execution
   epic (IBK-122..131) deliberately supersedes it for PAPER accounts;
   live trading remains out of scope.
2. **`optionsbot.ibkr` types stay inside the IBKR layer.** Downstream
   code imports adapter dataclasses, never `from ib_async`.
3. **Async at the top.** IBKR + Telegram + scheduler are async;
   analysis / strategies / scoring are sync.
4. **TDD.** Tests first, expect FAIL, then implement.
5. **One commit per task.** Body explains WHY in 1-3 paragraphs.

## Testing

```bash
uv run pytest -q             # full suite (default excludes live tests)
uv run pytest -m live        # live IBKR smoke test (requires IB Gateway)
uv run ruff check .          # lint
uv run mypy src              # type-check (strict mode on src/)
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `optionsbot status` ibkr fail | IB Gateway not running | Start IB Gateway on the configured port |
| Telegram 401 Unauthorized | Wrong bot_token | Re-check via @BotFather, update config.toml |
| Telegram 400 chat not found | You haven't messaged the bot | Send `/start` to your bot, re-run `optionsbot init --skip-telegram` |
| Daemon connects then disconnects | client_id collision (MCP also running on id 1) | Daemon uses id 2, MCP uses id 1; verify both with `settings.ibkr.client_id_*` |
| IV rank always 0.5 / warming_up | No daily ATM IV history yet | This is expected for v1 -- IV history collection is a deferred follow-up |
| Alerts never fire | Score threshold too high, or strategy doesn't match view | Lower `scan.score_threshold` or check `optionsbot status` for "last scan" |
| `optionsbot init` overwrites my config | You confirmed the overwrite prompt | Edit your config.toml; init is idempotent on re-run with `--non-interactive` |
| Market data errors in paper mode | Delayed data not available | `reqMarketDataType(3)` is called on connect; actual availability depends on your IBKR account state |

## Project status

v1: paper trading only, personal use. The scope is intentionally small
-- one user, one IBKR account, one Telegram chat. Multi-account and
live trading are explicitly out of scope for v1 (and v1 is the current
target). The auto-execution epic (IBK-122..IBK-131) is in progress:
paper-only automated order placement, off by default, semi-auto
(Telegram-confirmed) before any full-auto mode.

Implementation is tracked in YouTrack project
[IBK](https://tracker.example.invalid/projects/0-2); implementation
plans live in [`docs/superpowers/plans/`](docs/superpowers/plans/).

## License

Personal project. Not licensed for redistribution without permission.
