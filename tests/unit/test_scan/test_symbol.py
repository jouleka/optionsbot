"""Tests for scan_symbol."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from optionsbot.scan import ScanResult, scan_symbol
from optionsbot.storage.schema import snapshots, strategy_scores


async def test_scan_symbol_returns_scan_result(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    assert isinstance(result, ScanResult)
    assert result.symbol == "SPY"
    assert result.snapshot_id > 0
    assert result.view is not None
    assert isinstance(result.scored, tuple)


async def test_scan_symbol_persists_snapshot(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        rows = conn.execute(select(snapshots)).fetchall()
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"
    assert rows[0].spot == 400.0


async def test_scan_symbol_persists_strategy_scores(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        rows = conn.execute(
            select(strategy_scores).where(
                strategy_scores.c.snapshot_id == result.snapshot_id
            )
        ).fetchall()
    assert len(rows) >= 1
    for r in rows:
        assert 0.0 <= r.score <= 100.0
        assert r.legs_json is not None


async def test_scan_symbol_ensures_ibkr_connected(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    mock_ibkr_for_scan.ensure_connected.assert_awaited()  # type: ignore[union-attr]


async def test_scan_symbol_applies_view_override(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,  # type: ignore[arg-type]
        scan_engine,  # type: ignore[arg-type]
        scan_settings,  # type: ignore[arg-type]
        view_override=("bear", "high"),
    )
    assert result.view.direction == "bear"
    assert result.view.iv_regime == "high"


async def test_scan_symbol_partial_view_override_only_direction(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """Override one field; the other stays as inferred."""
    result_inferred = await scan_symbol(
        "SPY", mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[arg-type]
    )
    inferred_iv = result_inferred.view.iv_regime

    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,  # type: ignore[arg-type]
        scan_engine,  # type: ignore[arg-type]
        scan_settings,  # type: ignore[arg-type]
        view_override=("bear", None),
    )
    assert result.view.direction == "bear"
    assert result.view.iv_regime == inferred_iv


async def test_scan_symbol_normalizes_nan_hv20_to_none(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """historical_volatility returns NaN when bars are shorter than window+1;
    scan_symbol must normalize that to None so the snapshots row stores NULL
    and the scoring layer's `hv is None` guard catches it as intended.
    """
    # Replace the history fixture with one shorter than window=20.
    import optionsbot.scan.symbol as symbol_mod
    short_dates = [date(2026, 5, 1) + timedelta(days=i) for i in range(5)]
    short_bars = pd.DataFrame(
        {
            "open": [400.0] * 5,
            "high": [401.0] * 5,
            "low": [399.0] * 5,
            "close": [400.0 + i for i in range(5)],
            "volume": [1_000_000] * 5,
        },
        index=pd.Index(short_dates, name="date"),
    )
    short_history = symbol_mod.HistoryClient.return_value  # type: ignore[attr-defined]
    short_history.get_history.return_value = short_bars

    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(
            select(snapshots).where(snapshots.c.id == result.snapshot_id)
        ).first()
    assert row is not None
    # Persisted column is NULL (None when read back), NOT NaN.
    assert row.hv20 is None
    # iv_hv_ratio should also be None since hv20 collapsed to None.
    assert row.iv_hv_ratio is None


async def test_scan_symbol_passes_spot_and_strike_window_to_get_chain(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """scan_symbol feeds the underlying spot + configured strike window into
    get_chain so it can bound the fetch to near-ATM strikes."""
    import optionsbot.scan.symbol as symbol_mod

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    get_chain = symbol_mod.ChainClient.return_value.get_chain  # type: ignore[attr-defined]
    get_chain.assert_awaited_once()
    kwargs = get_chain.await_args.kwargs
    assert kwargs["underlying_price"] == 400.0  # fake_stock_quote.mid
    assert kwargs["strike_band_pct"] == scan_settings.scan.strike_band_pct  # type: ignore[attr-defined]
    assert kwargs["max_strikes_per_side"] == scan_settings.scan.max_strikes_per_side  # type: ignore[attr-defined]


async def test_scan_symbol_persists_earnings_in_window(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(select(snapshots)).fetchone()
    assert "earnings_in_window" in row.raw_json
    assert isinstance(row.raw_json["earnings_in_window"], bool)


async def test_scan_symbol_survives_news_failure(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    import optionsbot.scan.symbol as symbol_mod

    def _boom(*a: object, **k: object) -> None:
        raise RuntimeError("news down")

    monkeypatch.setattr(symbol_mod, "refresh_news_if_stale", _boom, raising=False)
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    assert result.snapshot_id > 0  # scan completed despite the news failure


async def test_scan_symbol_persists_relative_strength(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    # benchmark_symbol defaults to "SPY"; scanning SPY short-circuits to 0.0.
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(select(snapshots)).fetchone()
    assert "relative_strength" in row.raw_json
    assert row.raw_json["relative_strength"] == 0.0


async def test_scan_symbol_survives_relative_strength_failure(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    import optionsbot.scan.symbol as symbol_mod

    scan_settings.scan.benchmark_symbol = "QQQ"  # != SPY -> compute path runs  # type: ignore[attr-defined]

    def _boom(*a: object, **k: object) -> None:
        raise RuntimeError("benchmark down")

    monkeypatch.setattr(symbol_mod, "relative_strength", _boom)
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    assert result.snapshot_id > 0
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(
            select(snapshots).where(snapshots.c.id == result.snapshot_id)
        ).fetchone()
    assert row.raw_json["relative_strength"] is None


async def test_scan_symbol_skips_scoring_when_chain_has_no_option_data(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """If no chain leg has IV/greeks (e.g. missing OPRA option market data),
    scan_symbol must NOT run the scorers -- strategies built off the directional
    view alone would carry no real strikes/pricing and be misleading."""
    from typing import cast
    from unittest.mock import MagicMock

    import optionsbot.scan.symbol as symbol_mod
    from optionsbot.ibkr.types import OptionChainLeg, OptionRight

    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    nodata_chain = [
        OptionChainLeg(
            symbol="SPY", expiry=expiry, strike=k, right=cast("OptionRight", r),
            bid=None, ask=None, iv=None, delta=None, gamma=None,
            theta=None, vega=None, open_interest=None, volume=None,
        )
        for k in (400.0, 405.0, 410.0)
        for r in ("C", "P")
    ]
    symbol_mod.ChainClient.return_value.get_chain.return_value = nodata_chain  # type: ignore[attr-defined]
    spy_score = MagicMock(return_value=())
    monkeypatch.setattr(symbol_mod, "score_all", spy_score)

    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    spy_score.assert_not_called()
    assert result.scored == ()


async def test_scan_symbol_passes_line_cap_to_chain_client(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """scan_symbol must construct ChainClient with the configured
    max_market_data_lines so the streaming-line cap is honored."""
    import optionsbot.scan.symbol as symbol_mod

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    ctor_kwargs = symbol_mod.ChainClient.call_args.kwargs  # type: ignore[attr-defined]
    assert ctor_kwargs["max_market_data_lines"] == scan_settings.ibkr.max_market_data_lines  # type: ignore[attr-defined]


async def test_scan_symbol_passes_account_value_and_risk_pct_to_score_all(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """scan_symbol derives account_value from net_liquidation and passes it
    plus the configured risk_pct into score_all."""
    from unittest.mock import MagicMock

    import optionsbot.scan.symbol as symbol_mod

    spy_score = MagicMock(return_value=())
    monkeypatch.setattr(symbol_mod, "score_all", spy_score)

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    spy_score.assert_called_once()
    kwargs = spy_score.call_args.kwargs
    assert kwargs["account_value"] == 100000.0
    assert kwargs["risk_pct"] == scan_settings.scan.risk_pct  # type: ignore[attr-defined]


async def test_scan_symbol_account_value_none_when_no_net_liquidation(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """When the account summary has no net_liquidation, account_value is None
    (position sizing skipped)."""
    from unittest.mock import MagicMock

    import optionsbot.scan.symbol as symbol_mod
    from optionsbot.ibkr.types import AccountSummary

    symbol_mod.PositionsClient.return_value.get_account_summary.return_value = (  # type: ignore[attr-defined]
        AccountSummary(
            net_liquidation=None, buying_power=None, available_funds=None, currency="USD"
        )
    )
    spy_score = MagicMock(return_value=())
    monkeypatch.setattr(symbol_mod, "score_all", spy_score)

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    assert spy_score.call_args.kwargs["account_value"] is None


async def test_scan_symbol_records_atm_iv_history(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """A scan with real option data records today's ATM IV into iv_history."""
    from optionsbot.storage.iv_history import read_atm_iv_history

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    series = read_atm_iv_history(scan_engine, "SPY")  # type: ignore[arg-type]
    assert len(series) == 1
    assert series.iloc[0] == 0.20  # fake_chain ATM call iv


async def test_scan_symbol_iv_rank_active_after_warmup(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """With >=30 prior daily ATM-IV samples, iv_rank stops warming up and
    produces a real rank value."""
    from optionsbot.storage.iv_history import record_atm_iv

    for i in range(30):
        record_atm_iv(
            scan_engine,  # type: ignore[arg-type]
            "SPY",
            date(2026, 1, 1) + timedelta(days=i),
            0.15 + i * 0.005,
        )
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    assert result.view.warming_up is False
    assert result.view.iv_rank_value is not None


async def test_scan_symbol_no_iv_history_when_no_option_data(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """No option data (atm_iv=None) -> no iv_history row, warming_up stays True."""
    from typing import cast

    import optionsbot.scan.symbol as symbol_mod
    from optionsbot.ibkr.types import OptionChainLeg, OptionRight
    from optionsbot.storage.iv_history import read_atm_iv_history

    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    nodata_chain = [
        OptionChainLeg(
            symbol="SPY", expiry=expiry, strike=k, right=cast("OptionRight", r),
            bid=None, ask=None, iv=None, delta=None, gamma=None,
            theta=None, vega=None, open_interest=None, volume=None,
        )
        for k in (400.0, 405.0)
        for r in ("C", "P")
    ]
    symbol_mod.ChainClient.return_value.get_chain.return_value = nodata_chain  # type: ignore[attr-defined]

    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    assert result.view.warming_up is True
    assert len(read_atm_iv_history(scan_engine, "SPY")) == 0  # type: ignore[arg-type]


async def test_scan_symbol_passes_back_dte_gap_to_get_chain(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """scan_symbol fetches a back-month via settings.scan.back_month_dte_gap."""
    import optionsbot.scan.symbol as symbol_mod

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    kwargs = symbol_mod.ChainClient.return_value.get_chain.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["back_dte_gap"] == scan_settings.scan.back_month_dte_gap  # type: ignore[attr-defined]
