# optionsbot

[![CI](https://github.com/jouleka/optionsbot/actions/workflows/ci.yml/badge.svg)](https://github.com/jouleka/optionsbot/actions/workflows/ci.yml)
[![CodeQL](https://github.com/jouleka/optionsbot/actions/workflows/codeql.yml/badge.svg)](https://github.com/jouleka/optionsbot/actions/workflows/codeql.yml)
[![Secret scan](https://github.com/jouleka/optionsbot/actions/workflows/secret-scan.yml/badge.svg)](https://github.com/jouleka/optionsbot/actions/workflows/secret-scan.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental IBKR options research, alerting, and opt-in automated-execution tool for paper accounts.

> [!WARNING]
> This project is unaudited research software, not financial advice. It supports paper trading only.
> Order placement is disabled by default, paper fills can be unrealistically favorable, and no result
> should be treated as evidence of live profitability. Never point it at a live account.

## Safety model

- `execution.enabled` defaults to `false`.
- `execution.paper_only` defaults to `true` and refuses recognized live ports.
- A persisted kill switch, including Telegram `/kill`, halts execution across restarts.
- Every entry is subject to market-hours, freshness, risk, buying-power, portfolio-heat, and paper-account gates.
- Broker orders, executions, positions, and restart reconciliation are stored in a durable ledger.
- IBKR credentials are owned by IB Gateway or TWS and are never read by this application.

These controls reduce accidental misuse; they are not a guarantee. Review the source and configuration before running it.

## Features

- Scans configurable equity watchlists and option chains during market hours.
- Computes volatility measures, a market view, and scores 16 options strategies.
- Persists snapshots and outcomes to SQLite.
- Sends thresholded, deduplicated Telegram alerts.
- Exposes analysis, watchlist, position, and supervised-control tools over MCP.
- Supports opt-in Telegram-confirmed or automatic paper orders using atomic limit orders.
- Includes an optional 0DTE opening-range/fair-value-gap paper strategy with deterministic stops and targets.

The MCP process cannot mutate broker state; broker mutations remain in the daemon. Live-account routing is intentionally unsupported.

### Optional 0DTE opening-range strategy

The paper-only opening-range mode records the first 10 minutes after the New
York open on one-minute bars. After a close-confirmed breakout, it can recognize
either a fair-value-gap pullback or a rejection and retest of the broken range
boundary. Bullish and bearish breaks are evaluated independently so an early
false break does not suppress a later reversal.

Confirmed setups use defined-risk long options or debit spreads. Entry
admission requires a versioned, out-of-sample probability that the premium
target is observed before the stop or timeout, then subtracts estimated
round-trip costs. No such model is currently promoted, so managed expectancy is
unavailable and automatic entry fails closed. The daemon instead records every
confirmed candidate before alert or admission filtering and prospectively shadows
its executable bid/ask path. This produces target/stop/timeout/censored labels
without treating terminal expiry probability as intraday trade authority.

The restricted Hermes integration is an asynchronous, research-only context
critic. It can attach structured macro, news, event-conflict, or operational
observations to daemon-generated opportunities. It cannot originate a candidate,
review or authorize an entry, change strategy or risk settings, halt the bot, or
place an order. Its output is excluded from production admission artifacts.
The daemon remains the only component authorized to submit an order.

Model promotion is deliberately slower than model fitting: one eligible causal
base challenger is frozen at a time, then must pass a checksummed block of
strictly later sessions before it can authorize paper entries. A replacement
must also outperform the frozen incumbent on that same future block. Shadow
structure variants and Hermes context remain research-only and cannot be used
as order authority.

The generators, label reducer, and promotion gates have software tests; they do
not demonstrate a profitable strategy. No base model may trade until enough
prospective data exists and an out-of-sample artifact is explicitly promoted.

## Architecture

| Package | Responsibility |
|---|---|
| `optionsbot.config` | Typed configuration: environment, TOML, then defaults |
| `optionsbot.storage` | SQLAlchemy schema and Alembic migrations |
| `optionsbot.ibkr` | `ib_async` adapter and broker-facing types |
| `optionsbot.analysis` | Volatility and market-view calculations |
| `optionsbot.strategies` | Strategy registry and factor weights |
| `optionsbot.scoring` | Composite scoring, selection, and rationale |
| `optionsbot.scan` | End-to-end single-symbol scan |
| `optionsbot.alerts` | Telegram alert formatting |
| `optionsbot.mcp_server` | MCP analysis and supervised-control tools |
| `optionsbot.daemon` | Scheduled scans, alerts, and broker orchestration |
| `optionsbot.execution` | Safety gates, sizing, order lifecycle, reconciliation, and exits |
| `optionsbot.observability` | Structured logging and context propagation |

Analysis and scoring are synchronous pure logic; broker, scheduler, and Telegram boundaries are asynchronous. Broker-library types stay inside `optionsbot.ibkr`.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- IB Gateway or TWS connected to a paper account
- Optional Telegram bot for alerts and controls

## Install

```bash
git clone https://github.com/jouleka/optionsbot.git
cd optionsbot
uv sync --locked --group dev
uv run optionsbot init
```

The initializer writes `~/.config/optionsbot/config.toml`, applies database migrations, and can test Telegram. It is safe to rerun. For scripted setup, use `--non-interactive` and optionally `--skip-telegram`.

See [Telegram setup](docs/telegram-setup.md) for BotFather and chat ID instructions.

## Run

```bash
uv run optionsbot status
uv run optionsbot status --json
uv run optionsbot-daemon
```

For a generic user-service example, see [systemd setup](docs/systemd.md).
For a hardened paper-only VPS layout with headless IB Gateway, see the
[deployment templates](DEPLOYMENT.md). Copy the examples and keep all runtime
credentials outside the checkout.

### MCP configuration

Copy [`.mcp.json.example`](.mcp.json.example), replace `/path/to/optionsbot` with the absolute checkout path, and keep the resulting `.mcp.json` local. The trusted operator server exposes watchlist, analysis, snapshot, position, track-record, daily-briefing, candidate-review, close-request, and monotonic-halt tools. Do not give that endpoint to Hermes. The restricted Hermes endpoint exposes bounded ledger reads and the shadow context-critic contract; it has no proposal, entry-review, order, exit, halt, or rearm tool.

## Configuration

Settings live in `~/.config/optionsbot/config.toml`. Environment overrides use the `OPTIONSBOT_` prefix and a double underscore between section and field.

| Setting | Environment variable | Default |
|---|---|---|
| IBKR host | `OPTIONSBOT_IBKR__HOST` | `127.0.0.1` |
| IBKR port | `OPTIONSBOT_IBKR__PORT` | `4002` |
| Scan interval | `OPTIONSBOT_SCAN__INTERVAL_MINUTES` | `15` |
| Score threshold | `OPTIONSBOT_SCAN__SCORE_THRESHOLD` | `55` |
| Telegram token | `OPTIONSBOT_TELEGRAM__BOT_TOKEN` | unset |
| Telegram chat | `OPTIONSBOT_TELEGRAM__CHAT_ID` | unset |
| Database path | `OPTIONSBOT_STORAGE__DB_PATH` | `~/.local/share/optionsbot/optionsbot.db` |
| Execution switch | `OPTIONSBOT_EXECUTION__ENABLED` | `false` |
| Execution mode | `OPTIONSBOT_EXECUTION__MODE` | `confirm` |
| Paper-only interlock | `OPTIONSBOT_EXECUTION__PAPER_ONLY` | `true` |
| Managed shadow capture | `OPTIONSBOT_VALIDATION__MANAGED_CAPTURE_ENABLED` | `true` |
| Managed capture cadence | `OPTIONSBOT_VALIDATION__MANAGED_CAPTURE_INTERVAL_SECONDS` | `15` |
| Managed-model auto-promotion | `OPTIONSBOT_MANAGED_LEARNING__AUTO_PROMOTE` | `false` |

The managed outcome-policy identity is derived from label-affecting capture
settings, including polling cadence and phase. If an explicit policy identity
no longer matches those settings, configuration fails instead of pooling
incompatible labels.

Recognized default ports are 4002/7497 for paper and 4001/7496 for live. The MCP server and daemon use distinct client IDs (1 and 2 by default).

Copy `.env.example` only if environment-based configuration is useful. Never commit the resulting `.env`, API tokens, broker credentials, databases, or production host details.

## Development

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pip-audit
uv run --locked bandit -q -ll -r src
```

The default test command excludes tests marked `live`. `uv run pytest -m live` requires a running paper IB Gateway and may create paper orders; run it deliberately.

## Status and scope

The project is designed for one operator, one IBKR paper account, and one Telegram chat. Paper execution, order tracking, reconciliation, risk gates, deterministic exits, and optional external-review hooks are implemented. Execution remains off by default. Multi-account and live trading are out of scope.

The current 0DTE research architecture, managed-payoff formula, Hermes boundary,
and validation/promotion plan are documented in
[the strategy redesign](docs/strategy-redesign.md).

## Security

Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Do not include credentials or private infrastructure details in a public issue.

## License

[MIT](LICENSE)
