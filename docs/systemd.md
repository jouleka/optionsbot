# Running the daemon with systemd

Use a user service so the daemon runs without root privileges. Replace the checkout and `uv` paths below with paths from your own machine.

```ini
# ~/.config/systemd/user/optionsbot-daemon.service
[Unit]
Description=optionsbot paper-trading daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/optionsbot
ExecStart=/path/to/uv run --locked optionsbot-daemon
Restart=on-failure
RestartSec=10
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
```

Then load and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now optionsbot-daemon.service
systemctl --user status optionsbot-daemon.service
journalctl --user -u optionsbot-daemon.service -f
```

Keep configuration and credentials outside the repository. Verify `execution.enabled=false` and `execution.paper_only=true` before the first start. Do not run this service against a live IBKR account.
