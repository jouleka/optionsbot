#!/usr/bin/python3
"""Root launcher that passes exactly one provider secret, then drops privilege.

Install this file root-owned and non-writable at
``/usr/local/bin/optionsbot-market-context-launcher``. A trusted external-review
process invokes it with either ``fred`` or ``finnhub``; secret values never
appear in argv or config.
"""

from __future__ import annotations

import os
import pwd
import stat
import sys
from pathlib import Path

SECRET_FILE = Path("/etc/optionsbot/market-context.env")
INSTALL_ROOT = Path("/opt/optionsbot-market-context")
PROVIDERS = {
    "fred": ("optionsbot-fred", "FRED_API_KEY", "optionsbot-fred-mcp"),
    "finnhub": ("optionsbot-finnhub", "FINNHUB_API_KEY", "optionsbot-finnhub-mcp"),
}


def _read_secret(name: str) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(SECRET_FILE, flags)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError("market-context secret file ownership or mode is unsafe")
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key == name:
                    secret = value.strip()
                    if not secret:
                        break
                    return secret
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raise RuntimeError(f"required credential {name} is not configured")


def main() -> int:
    if os.geteuid() != 0:
        raise RuntimeError("market-context launcher must start as root")
    if len(sys.argv) != 2 or sys.argv[1] not in PROVIDERS:
        raise RuntimeError("usage: optionsbot-market-context-launcher {fred|finnhub}")

    provider = sys.argv[1]
    username, secret_name, executable_name = PROVIDERS[provider]
    account = pwd.getpwnam(username)
    executable = INSTALL_ROOT / ".venv" / "bin" / executable_name
    home = Path(account.pw_dir)
    if not executable.is_file():
        raise RuntimeError("market-context executable is not installed")

    secret = _read_secret(secret_name)
    os.chdir(home)
    os.umask(0o077)
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    environment = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "PYTHONUNBUFFERED": "1",
        secret_name: secret,
    }
    executable_text = str(executable)
    os.execve(executable_text, [executable_text], environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
