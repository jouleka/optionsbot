# OptionsBot paper VPS example

This runbook installs OptionsBot and a headless IB Gateway on a systemd Linux
host. It assumes a dedicated non-root `optionsbot` account and the checkout at
`/home/optionsbot/optionsbot`. Adjust paths consistently if your layout differs.

> [!WARNING]
> The included profile can automatically create paper orders. It cannot make
> paper fills realistic and is not evidence of profitability. Keep the
> paper-account interlock enabled and never use these units with a live account.

No credential belongs in this repository. The runtime files described below
must remain outside the checkout and mode `0600`.

## 1. Provision the service account

Run as root on a supported Ubuntu or Debian host:

```bash
apt-get update
apt-get install -y curl git unzip xvfb
id optionsbot >/dev/null 2>&1 || useradd -m -s /bin/bash optionsbot
```

## 2. Install the application

```bash
sudo -u optionsbot -H bash -lc '
  command -v ~/.local/bin/uv >/dev/null 2>&1 || \
    curl -LsSf https://astral.sh/uv/install.sh | sh
  git clone https://github.com/jouleka/optionsbot.git ~/optionsbot
  cd ~/optionsbot
  git switch main
  ~/.local/bin/uv sync --locked --group dev
'
```

For later updates, fast-forward only from the public default branch:

```bash
sudo -u optionsbot -H bash -lc '
  cd ~/optionsbot
  git fetch origin main
  git merge --ff-only origin/main
  ~/.local/bin/uv sync --locked --group dev
'
```

## 3. Install IB Gateway and IBC

Install the official IB Gateway paper client and
[IBC](https://github.com/IbcAlpha/IBC) following their current upstream
instructions. The provided launcher expects:

- IB Gateway below `~optionsbot/Jts/ibgateway/<numeric-version>`;
- IBC below `/opt/ibc`;
- the private IBC config at `~optionsbot/ibc/config.ini`.

Create the private config from the template, replace both `CHANGEME` values,
and calculate `AutoRestartTime` in the server's local time zone so it falls
outside US market hours:

```bash
sudo -u optionsbot -H bash -lc '
  install -d -m 700 ~/ibc ~/.local/bin
  install -m 600 ~/optionsbot/deploy/gateway/config.ini.template ~/ibc/config.ini
  install -m 755 ~/optionsbot/deploy/gateway/optionsbot-gateway-start.sh \
    ~/.local/bin/optionsbot-gateway-start.sh
  ${EDITOR:-vi} ~/ibc/config.ini
'
```

## 4. Create private OptionsBot configuration

The example profile is intentionally paper-only but enables automatic paper
entries. Review every risk and timing value before using it:

```bash
sudo -u optionsbot -H bash -lc '
  install -d -m 700 ~/.config/optionsbot
  install -m 600 \
    ~/optionsbot/deploy/vps/config.paper.example.toml \
    ~/.config/optionsbot/config.toml
  ${EDITOR:-vi} ~/.config/optionsbot/config.toml
'
```

Add Telegram settings only to the private runtime file if alerts or controls
are required:

```toml
[telegram]
bot_token = "replace-locally"
chat_id = "replace-locally"
```

Validate the paper interlock without printing secrets:

```bash
sudo -u optionsbot -H bash -lc 'cd ~/optionsbot && ./.venv/bin/python - <<"PY"
from pathlib import Path
from optionsbot.config import PAPER_PORTS, load_settings

settings = load_settings(Path.home() / ".config/optionsbot/config.toml")
assert settings.execution.paper_only
assert settings.ibkr.port in PAPER_PORTS
print("CONFIG OK: paper-only port and interlock")
PY'
```

Apply migrations before starting the daemon:

```bash
sudo -u optionsbot -H bash -lc 'cd ~/optionsbot && ./.venv/bin/optionsbot migrate'
```

## 5. Optional issue-tracker reporter

The soak reporter is optional. If enabled, keep its credential isolated from
the unprivileged daemon in a dedicated root-owned file:

```bash
install -d -o root -g root -m 700 /etc/optionsbot
install -o root -g root -m 600 \
  /home/optionsbot/optionsbot/deploy/vps/reporter.env.example \
  /etc/optionsbot/reporter.env
${EDITOR:-vi} /etc/optionsbot/reporter.env
```

Leave `optionsbot-soak-reporter.timer` disabled when no reporter is configured.

## 6. Install systemd units

```bash
repo=/home/optionsbot/optionsbot
install -m 644 \
  "$repo/deploy/vps/ibc-xvfb.service" \
  "$repo/deploy/vps/optionsbot-gateway.service" \
  "$repo/deploy/vps/optionsbot-daemon.service" \
  "$repo/deploy/vps/optionsbot-daemon-failure.service" \
  "$repo/deploy/vps/optionsbot-rth-acceptance.service" \
  "$repo/deploy/vps/optionsbot-rth-acceptance.timer" \
  "$repo/deploy/vps/optionsbot-soak-reporter.service" \
  "$repo/deploy/vps/optionsbot-soak-reporter.timer" \
  /etc/systemd/system/
install -m 700 \
  "$repo/deploy/vps/optionsbot-soak-reporter-launcher" \
  "$repo/deploy/vps/optionsbot-venv-ownership-guard" \
  /usr/local/libexec/
install -m 755 "$repo/deploy/vps/optionsbot-mcp-restricted" /usr/local/bin/

systemctl daemon-reload
systemctl enable --now ibc-xvfb optionsbot-gateway
# Wait for the paper API to become available, then:
systemctl enable --now optionsbot-daemon optionsbot-rth-acceptance.timer
# Enable the reporter only after /etc/optionsbot/reporter.env is configured:
# systemctl enable --now optionsbot-soak-reporter.timer
```

## 7. Verify and operate

```bash
systemctl status optionsbot-gateway optionsbot-daemon
systemctl list-timers optionsbot-rth-acceptance.timer optionsbot-soak-reporter.timer
journalctl -u optionsbot-gateway -n 100 --no-pager
journalctl -u optionsbot-daemon -n 100 --no-pager
sudo -u optionsbot -H bash -lc 'cd ~/optionsbot && ./.venv/bin/optionsbot status'
```

Before restarting after an update, compare installed units with the checked-in
templates and verify that the private IBC and OptionsBot configurations still
have mode `0600`. Restart only the affected services.

The acceptance timer records a protected session ledger under
`~optionsbot/.local/state/optionsbot/soak_evidence.json`. It reports paper
health and recovery evidence; it does not change trading mode, close an issue,
or authorize a live order.
