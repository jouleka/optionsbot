"""screen_universe orchestration against a fake history client (IBK-95)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optionsbot.screener.screen import screen_universe


class _FakeHistory:
    """Returns canned bars per symbol; raises for symbols in ``fail``."""

    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame], fail: set[str]) -> None:
        self._bars = bars_by_symbol
        self._fail = fail

    async def get_history(self, symbol: str, days: int = 252) -> pd.DataFrame:
        if symbol in self._fail:
            raise ValueError(f"no data for {symbol}")
        return self._bars[symbol]


def _bars(seed: int, vol: float, volume: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, vol, 120))
    return pd.DataFrame({"close": closes, "volume": np.full(120, volume)})


async def test_screen_universe_ranks_and_skips_failures() -> None:
    bars = {
        "AAA": _bars(1, 0.03, 2_000_000.0),  # liquid
        "BBB": _bars(2, 0.01, 2_000_000.0),  # liquid
        "ILQ": _bars(3, 0.05, 1.0),          # illiquid -> gated out
    }
    fake = _FakeHistory(bars, fail={"ZZZ"})
    out = await screen_universe(fake, ["AAA", "BBB", "ILQ", "ZZZ"], min_dollar_volume=1_000_000.0)  # type: ignore[arg-type]
    syms = [c.symbol for c in out]
    assert "ILQ" not in syms  # gated by liquidity
    assert "ZZZ" not in syms  # fetch failed -> skipped
    assert set(syms) == {"AAA", "BBB"}
    assert all(0.0 <= c.hv_rank <= 1.0 for c in out)
