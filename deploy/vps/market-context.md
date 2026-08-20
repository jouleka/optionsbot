# Optional market-context MCP deployment

This deployment gives an external-review profile additional analysis context without
giving either provider process access to IBKR, the Optionsbot database, Telegram, or
the other provider credential.

## Security boundary

- `optionsbot-fred` receives only `FRED_API_KEY` and exposes two fixed-host, allowlisted
  numeric tools.
- `optionsbot-finnhub` receives only `FINNHUB_API_KEY` and exposes quote, company-news,
  and earnings-calendar tools from verified free-tier endpoints.
- Secrets are loaded by a root-owned launcher from
  `/etc/optionsbot/market-context.env`, which must have mode `0600`. They do
  not appear in the review-agent config or process command lines.
- Both accounts are non-login users. Their state directories have mode `0700` and no
  permission to read the trading runtime or ledger.
- Finnhub prose is capped, control-character sanitized, and explicitly labeled untrusted.
  External context is analysis input only and never authorizes an entry or bypasses the
  daemon's independent exit gates.

## Install

Run from a trusted checkout after tests pass. Replace `/path/to/optionsbot`
with that checkout's absolute path:

```bash
useradd --system --home-dir /var/lib/optionsbot-fred --create-home --shell /usr/sbin/nologin optionsbot-fred
useradd --system --home-dir /var/lib/optionsbot-finnhub --create-home --shell /usr/sbin/nologin optionsbot-finnhub
chmod 700 /var/lib/optionsbot-fred /var/lib/optionsbot-finnhub

rm -rf /opt/optionsbot-market-context
python3 -m venv /opt/optionsbot-market-context/.venv
/opt/optionsbot-market-context/.venv/bin/pip install /path/to/optionsbot
chown -R root:root /opt/optionsbot-market-context
chmod -R go-w /opt/optionsbot-market-context
install -o root -g root -m 700 \
  deploy/vps/optionsbot-market-context-launcher.py \
  /usr/local/bin/optionsbot-market-context-launcher
```

Create `/etc/optionsbot/market-context.env` as a root-owned mode-`0600` file
containing only `FRED_API_KEY` and `FINNHUB_API_KEY`. Then mount the launchers
as distinct stdio MCP servers in the external-review profile and use explicit
tool includes:

```yaml
mcp_servers:
  fred:
    command: /usr/local/bin/optionsbot-market-context-launcher
    args: [fred]
    tools:
      include: [fred_macro_snapshot, fred_series]
      prompts: false
      resources: false
  finnhub:
    command: /usr/local/bin/optionsbot-market-context-launcher
    args: [finnhub]
    tools:
      include: [finnhub_quote, finnhub_company_news, finnhub_earnings_calendar]
      prompts: false
      resources: false
```

Restart the external-review service, inspect its MCP stderr log, and exercise
one tool from each server. Do not restart the trading daemon or IB Gateway for
this change.
