"""`optionsbot screen` CLI command (IBK-95)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from optionsbot.cli import app
from optionsbot.screener.screen import ScreenCandidate

runner = CliRunner()


def test_screen_lists_ranked_candidates() -> None:
    candidates = (
        ScreenCandidate(symbol="NVDA", hv_rank=0.91, dollar_volume=42_000_000_000.0),
        ScreenCandidate(symbol="AAPL", hv_rank=0.55, dollar_volume=9_000_000_000.0),
    )
    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()
    with patch("optionsbot.ibkr.IBKRClient", return_value=fake_client), patch(
        "optionsbot.screener.screen.screen_universe", new=AsyncMock(return_value=candidates)
    ):
        result = runner.invoke(app, ["screen", "--top", "1"])
    assert result.exit_code == 0
    assert "NVDA" in result.stdout
    assert "AAPL" not in result.stdout  # --top 1 cut it
