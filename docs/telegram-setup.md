# Telegram bot setup

The daemon's alerts go through your own private Telegram bot. Setup
takes about 3 minutes.

## 1. Create the bot via @BotFather

In Telegram, search for `@BotFather` and start a chat with it. Run:

```
/newbot
```

BotFather asks for a display name and a username (must end in `bot`).
At the end it prints a token like `123456:ABC-def...`. Copy it.

## 2. Get your chat_id

Send `/start` to the bot you just created (find it by its username in
Telegram). Then in a browser open:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

The response JSON contains your `message.chat.id` — a number like
`123456789`. That's your `chat_id`.

## 3. Configure optionsbot

Add to `~/.config/optionsbot/config.toml` (or as env vars):

```toml
[telegram]
bot_token = "123456:ABC-def..."
chat_id = "123456789"
```

Or env vars:

```bash
export OPTIONSBOT_TELEGRAM__BOT_TOKEN="123456:ABC-def..."
export OPTIONSBOT_TELEGRAM__CHAT_ID="123456789"
```

Note the double underscore (`__`) between section and field.

## 4. Verify

Run the daemon for a few seconds — it logs `TelegramClient` errors at
startup if either is missing. Or run a one-off Python check:

```bash
uv run python -c "
import asyncio
from optionsbot.config import get_settings
from optionsbot.daemon.telegram_client import TelegramClient

async def go():
    s = get_settings()
    tg = TelegramClient(s.telegram.bot_token, s.telegram.chat_id)
    msg_id = await tg.send_message('optionsbot test message')
    print('sent message_id', msg_id)
    await tg.aclose()

asyncio.run(go())
"
```

You should receive the message in your Telegram chat within a couple
of seconds, and the script prints the `message_id`.

## What the daemon sends

For each ticker that crosses the alert threshold (default 70 score),
you get a Markdown message like:

```
SPY — iron_condor score 85.0
view: neutral/high (rank 0.72)

legs:
  • sell 20260711 410C
  • buy 20260711 415C
  • sell 20260711 390P
  • buy 20260711 385P

net credit $1.25
max loss $3.75
prob profit 68%
size 5 contracts

High IV rank + tight liquidity.
```

Undefined-risk strategies (Short Straddle, Short Strangle) prefix the
message with `⚠ UNDEFINED RISK`.

## Retry semantics

Telegram send failures are persisted in the `alerts` table with
exponential backoff: 1m, 5m, 15m, 60m, 240m. After 5 failed attempts
the alert is marked `dropped`. Each scheduler tick (15 minutes by
default) runs `sweep_retries` before the watchlist scan, so a transient
Telegram outage gets re-attempted at scheduler granularity until either
success or drop.

## Dedup

Within `alert_cooldown_hours` (default 4) of the last `sent` alert for
the same `(symbol, strategy)` pair, no new alert fires unless the score
jumped by more than `alert_rescore_delta` (default 10). Set
`alert_cooldown_hours=0` to disable the cooldown entirely.
