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

### MCP configuration

Copy [`.mcp.json.example`](.mcp.json.example), replace `/path/to/optionsbot` with the absolute checkout path, and keep the resulting `.mcp.json` local. The server exposes watchlist, analysis, snapshot, position, track-record, daily-briefing, candidate-review, close-request, and monotonic-halt tools.

## Configuration

Settings live in `~/.config/optionsbot/config.toml`. Environment overrides use the `OPTIONSBOT_` prefix and a double underscore between section and field.

| Setting | Environment variable | Default |
|---|---|---|
| IBKR host | `OPTIONSBOT_IBKR__HOST` | `127.0.0.1` |
| IBKR port | `OPTIONSBOT_IBKR__PORT` | `4002` |
| Scan interval | `OPTIONSBOT_SCAN__INTERVAL_MINUTES` | `15` |
| Score threshold | `OPTIONSBOT_SCAN__SCORE_THRESHOLD` | `70` |
| Telegram token | `OPTIONSBOT_TELEGRAM__BOT_TOKEN` | unset |
| Telegram chat | `OPTIONSBOT_TELEGRAM__CHAT_ID` | unset |
| Database path | `OPTIONSBOT_STORAGE__DB_PATH` | `~/.local/share/optionsbot/optionsbot.db` |
| Execution switch | `OPTIONSBOT_EXECUTION__ENABLED` | `false` |
| Execution mode | `OPTIONSBOT_EXECUTION__MODE` | `confirm` |
| Paper-only interlock | `OPTIONSBOT_EXECUTION__PAPER_ONLY` | `true` |

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

## Security

Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Do not include credentials or private infrastructure details in a public issue.

## License

[MIT](LICENSE)
