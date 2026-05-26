# optionsbot

Personal IBKR options-analysis tool. Paper-trading only for v1. Analysis and alerts; never places orders.

Implementation is tracked in YouTrack project [IBK](https://tracker.example.invalid/projects/0-2) and detailed plans live in `docs/superpowers/plans/`.

## Secrets

Copy `.env.example` to `.env` and fill in:
- `OPTIONSBOT_TELEGRAM__BOT_TOKEN` — from @BotFather in Telegram.
- `OPTIONSBOT_TELEGRAM__CHAT_ID` — your numeric chat ID (send `/start` to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates`).

The double underscore is intentional: it's the `env_nested_delimiter` configured on `Settings`, mapping `OPTIONSBOT_TELEGRAM__BOT_TOKEN` to `settings.telegram.bot_token`. Bare names like `TELEGRAM_BOT_TOKEN` will be silently ignored.

IBKR credentials are deliberately NOT in this repo. IB Gateway holds them.

Future: OS keyring (`keyring` Python package) is a possible v2 store for the Telegram token. Not v1.
