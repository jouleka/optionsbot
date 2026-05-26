# optionsbot

Personal IBKR options-analysis tool. Paper-trading only for v1. Analysis and alerts; never places orders.

Implementation is tracked in YouTrack project [IBK](https://tracker.example.invalid/projects/0-2) and detailed plans live in `docs/superpowers/plans/`.

## IB Gateway setup

This codebase does NOT auto-install or auto-launch IB Gateway. You must run the IB Gateway (or TWS) app yourself before the daemon or MCP server can connect.

Default ports (community convention, not exhaustively documented in IBKR's official TWS-API docs; override `settings.ibkr.port` in `~/.config/optionsbot/config.toml` if your install uses a different port):

| App        | Paper | Live |
|------------|-------|------|
| IB Gateway | 4002  | 4001 |
| TWS        | 7497  | 7496 |

In paper mode (default), the client calls `reqMarketDataType(3)` on connect to opt into delayed-streaming data. This typically avoids "no market data subscription" errors, but the actual availability of delayed data still depends on your IBKR account state.

Distinct `client_id` values are reserved per process role so the MCP server and the daemon can hold simultaneous connections without colliding: `settings.ibkr.client_id_mcp` (default 1) and `settings.ibkr.client_id_daemon` (default 2).

## Secrets

Copy `.env.example` to `.env` and fill in:
- `OPTIONSBOT_TELEGRAM__BOT_TOKEN` — from @BotFather in Telegram.
- `OPTIONSBOT_TELEGRAM__CHAT_ID` — your numeric chat ID (send `/start` to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates`).

The double underscore is intentional: it's the `env_nested_delimiter` configured on `Settings`, mapping `OPTIONSBOT_TELEGRAM__BOT_TOKEN` to `settings.telegram.bot_token`. Bare names like `TELEGRAM_BOT_TOKEN` will be silently ignored.

IBKR credentials are deliberately NOT in this repo. IB Gateway holds them.

Future: OS keyring (`keyring` Python package) is a possible v2 store for the Telegram token. Not v1.
