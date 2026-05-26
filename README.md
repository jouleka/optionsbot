# optionsbot

Personal IBKR options-analysis tool. Paper-trading only for v1. Analysis and alerts; never places orders.

Implementation is tracked in YouTrack project [IBK](https://tracker.example.invalid/projects/0-2) and detailed plans live in `docs/superpowers/plans/`.

## Secrets

Copy `.env.example` to `.env` and fill in:
- `TELEGRAM_BOT_TOKEN` — from @BotFather in Telegram.
- `TELEGRAM_CHAT_ID` — your numeric chat ID (send `/start` to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates`).

IBKR credentials are deliberately NOT in this repo. IB Gateway holds them.

Future: OS keyring (`keyring` Python package) is a possible v2 store for the Telegram token. Not v1.
