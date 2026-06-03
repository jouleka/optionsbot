"""Historical daily bars with on-disk parquet cache.

``HistoryClient.get_history`` returns a pandas DataFrame of OHLCV daily
bars for an equity symbol. Bars come from ``ib_async.reqHistoricalDataAsync``;
results are cached as parquet keyed by ``{symbol}-{end_date}.parquet`` so
intra-day reuse for the same window is free.

A cache hit returns the trailing ``days`` rows of the on-disk DataFrame.
A cache miss (or insufficient cached rows for the requested window) does
a fresh fetch and rewrites the parquet file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "optionsbot" / "history"


class HistoryClient:
    def __init__(
        self,
        client: IBKRClient,
        resolver: ContractResolver | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver if resolver is not None else ContractResolver(client)
        self._cache_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR

    def _cache_path(self, symbol: str, end_date: date) -> Path:
        return self._cache_dir / f"{symbol}-{end_date.isoformat()}.parquet"

    async def get_history(
        self,
        symbol: str,
        days: int = 252,
        end_date: date | None = None,
        duration_str: str | None = None,
    ) -> pd.DataFrame:
        end_date = end_date if end_date is not None else date.today()
        cache_path = self._cache_path(symbol, end_date)
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if len(cached) >= days:
                return cached.iloc[-days:]
            # Insufficient rows cached -- fall through and re-fetch a larger window.
        contract = await self._resolver.stock(symbol)
        await self._client.ensure_connected()
        bars = await self._client.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration_str if duration_str is not None else f"{days} D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            raise ValueError(f"No historical bars returned for symbol={symbol!r}")
        df = pd.DataFrame(
            {
                "date": [b.date for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            }
        )
        df = df.set_index("date").sort_index()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        return df
