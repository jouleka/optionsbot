# Deployment templates

The `deploy/` directory contains sanitized examples for running OptionsBot on
a systemd Linux host with a dedicated service account. It includes:

- IB Gateway/IBC launch templates;
- systemd units and helper launchers;
- an explicitly paper-only example configuration;
- optional Hermes prompts, market-context launchers, and a send-only Telegram
  adapter.

These files document one reproducible layout. They are examples, not a turnkey
production environment, and they intentionally contain no host addresses,
account identifiers, credentials, chat IDs, private tracker URLs, or tokens.

## Secret boundary

Never commit broker passwords, Telegram tokens or chat IDs, issue-tracker
tokens, API keys, SQLite databases, logs, or generated evidence. Keep runtime
values in owner-only files outside the checkout:

- `~optionsbot/ibc/config.ini` for the paper IB Gateway login;
- `~optionsbot/.config/optionsbot/config.toml` for OptionsBot and Telegram;
- `/etc/optionsbot/reporter.env` for the optional soak reporter;
- `/etc/optionsbot/market-context.env` for optional external-data providers.

The examples use placeholders for every credential. Secret-bearing runtime
files should be mode `0600` and must remain untracked.

## Updating a deployment

1. Review and merge application changes in this repository.
2. Run Ruff, MyPy, Pytest, dependency audit, Bandit, and Gitleaks.
3. Fast-forward the deployment checkout from public `main`.
4. Compare local runtime configuration and installed units before restarting.
5. Restart only the affected services and verify their health and paper-account
   interlocks.

See [`deploy/vps/README.md`](deploy/vps/README.md) for the example layout and
installation steps.
