# Running optionsbot-daemon under systemd --user

The daemon is a long-running async process. systemd --user gives you:
- auto-restart on failure
- journalctl-based logs
- start/stop via `systemctl --user`

## One-time setup

```bash
# 1. Install the unit file into your user systemd directory
mkdir -p ~/.config/systemd/user
cp packaging/systemd/optionsbot-daemon.service ~/.config/systemd/user/

# 2. Edit ExecStart to point at YOUR uv binary + project path
$EDITOR ~/.config/systemd/user/optionsbot-daemon.service
# replace `%h/.local/bin/uv` if `which uv` returns a different path
# replace `%h/projects/optionsbot` if your checkout is elsewhere

# 3. Reload systemd, then enable + start
systemctl --user daemon-reload
systemctl --user enable --now optionsbot-daemon.service
```

## Day-to-day

```bash
# View live logs (Ctrl-C to detach)
journalctl --user -u optionsbot-daemon -f

# View the last N lines without following
journalctl --user -u optionsbot-daemon -n 100

# Check service status
systemctl --user status optionsbot-daemon

# Stop / start / restart
systemctl --user stop optionsbot-daemon
systemctl --user start optionsbot-daemon
systemctl --user restart optionsbot-daemon

# After a code change, pull, then restart
git pull
systemctl --user restart optionsbot-daemon
```

## WSL2 prerequisite

WSL2 doesn't enable systemd by default. To turn it on:

```bash
# In WSL
sudo tee -a /etc/wsl.conf <<'EOF'
[boot]
systemd=true
EOF

# Back in Windows PowerShell
wsl --shutdown

# Reopen WSL; verify systemd is running
systemctl --user status
```

If `systemctl --user` returns "Failed to connect to bus", systemd-user
isn't running -- check the WSL conf above.

## Disable / uninstall

```bash
systemctl --user disable --now optionsbot-daemon
rm ~/.config/systemd/user/optionsbot-daemon.service
systemctl --user daemon-reload
```

## Why systemd --user and not system-wide?

The daemon connects to YOUR IB Gateway with YOUR client_id and stores
data in YOUR home directory. Running it under a system unit would
require running IB Gateway as root or worse -- a user unit keeps
everything in user-space, which matches the security model the rest
of optionsbot assumes.
