"""Tests for the optionsbot-daemon entry-point arg handling (IBK-98)."""

from __future__ import annotations

import pytest

from optionsbot.config import Settings
from optionsbot.daemon.__main__ import main


def test_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "optionsbot-daemon" in capsys.readouterr().out


def test_unknown_arg_exits_two() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--bogus"])
    assert exc.value.code == 2


def test_no_args_starts_daemon(monkeypatch) -> None:
    """With no args, main() parses cleanly and runs the daemon (mocked)."""
    monkeypatch.setattr("optionsbot.daemon.__main__.get_settings", lambda: Settings())
    ran: dict[str, bool] = {}

    def _fake_run(coro):
        ran["v"] = True
        coro.close()  # avoid 'coroutine never awaited' warning
        return 0

    monkeypatch.setattr("optionsbot.daemon.__main__.asyncio.run", _fake_run)
    assert main([]) == 0
    assert ran["v"] is True
