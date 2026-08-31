"""Tests for scan_symbol."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from sqlalchemy import select

from optionsbot.scan import ScanResult, scan_symbol
from optionsbot.storage.schema import snapshots, strategy_scores, symbol_news


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
    assert rows[0].expected_move == pytest.approx(
        400.0 * 0.20 * math.sqrt(45 / 365.0)
    )
    assert rows[0].raw_json["front_dte"] == 45
    assert rows[0].raw_json["expected_move"] == pytest.approx(rows[0].expected_move)


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


async def test_scan_symbol_sizes_with_usd_equity_for_non_usd_account(
    monkeypatch, mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:  # type: ignore[no-untyped-def]
    from optionsbot.ibkr.types import AccountSummary
    from optionsbot.scan import symbol as symbol_mod

    positions = MagicMock()
    positions.get_positions = AsyncMock(return_value=[])
    positions.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=Decimal("8000"),
            buying_power=Decimal("8000"),
            available_funds=Decimal("8000"),
            currency="EUR",
            fx_to_usd=Decimal("1.25"),
        )
    )
    monkeypatch.setattr(symbol_mod, "PositionsClient", MagicMock(return_value=positions))
    score_all = MagicMock(return_value=())
    monkeypatch.setattr(symbol_mod, "score_all", score_all)

    await scan_symbol(
        "SPY", mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[arg-type]
    )

    assert score_all.call_args.kwargs["account_value"] == 10_000.0


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
    assert kwargs["dte_window"] == (
        scan_settings.scan.dte_window_min,  # type: ignore[attr-defined]
        scan_settings.scan.dte_window_max,  # type: ignore[attr-defined]
    )
    assert kwargs["dte_target"] == scan_settings.scan.dte_target  # type: ignore[attr-defined]


async def test_scan_symbol_persists_earnings_in_window(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(select(snapshots)).fetchone()
    assert "earnings_in_window" in row.raw_json
    assert isinstance(row.raw_json["earnings_in_window"], bool)
    assert "next_earnings_date" in row.raw_json
    assert "earnings_source" in row.raw_json
    assert row.raw_json["beta_to_benchmark"] == 1.0
    assert row.raw_json["beta_benchmark"] == "SPY"


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


async def test_scan_symbol_prefers_and_persists_ibkr_api_news(
    mock_ibkr_for_scan: object,
    scan_engine: object,
    scan_settings: object,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    import optionsbot.scan.symbol as symbol_mod

    api_news = [
        {
            "title": "SPY catalyst",
            "publisher": "Dow Jones",
            "published_ts": "2026-07-27T17:44:00+00:00",
            "link": None,
            "source": "IBKR_API_NEWS",
            "provider_code": "DJ-N",
            "article_id": "DJ-N$1",
        }
    ]
    news_client = MagicMock()
    news_client.headlines = AsyncMock(return_value=api_news)
    monkeypatch.setattr(symbol_mod, "NewsClient", MagicMock(return_value=news_client))
    yahoo_fallback = MagicMock()
    monkeypatch.setattr(symbol_mod, "refresh_news_if_stale", yahoo_fallback)

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(select(symbol_news)).one()
    assert row.headlines_json == api_news
    yahoo_fallback.assert_not_called()


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

    # benchmark != the scanned symbol -> the compute path runs (not the 0.0 short-circuit)
    scan_settings.scan.benchmark_symbol = "QQQ"  # type: ignore[attr-defined]

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


async def test_scan_symbol_persists_real_relative_strength(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """Scan a NON-benchmark symbol so the REAL relative_strength compute runs
    (not the 0.0 short-circuit) -- guards arg order + window plumbing."""
    from unittest.mock import AsyncMock

    import optionsbot.scan.symbol as symbol_mod

    scan_settings.scan.relative_strength_window = 5  # type: ignore[attr-defined]

    def _frame(closes: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        })

    flat = [400.0] * 114
    sym_frame = _frame(flat + [400, 404, 408, 412, 416, 440])    # +10% over last 5
    bench_frame = _frame(flat + [400, 400, 404, 404, 408, 408])  # +2%

    async def _get_history(sym: str, *a: object, **k: object) -> pd.DataFrame:
        return bench_frame if sym == "SPY" else sym_frame

    symbol_mod.HistoryClient.return_value.get_history = AsyncMock(side_effect=_get_history)  # type: ignore[attr-defined]

    result = await scan_symbol("AAPL", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(
            select(snapshots).where(snapshots.c.id == result.snapshot_id)
        ).fetchone()
    rs = row.raw_json["relative_strength"]
    assert rs is not None
    assert round(rs, 4) == round(0.10 - 0.02, 4)  # +8%


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


async def test_orb_scan_persists_managed_ev_and_retains_terminal_ev(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    from optionsbot.analysis.opening_range_fvg import detect_opening_range_fvg
    from optionsbot.analysis.types import MarketView
    from optionsbot.scan import symbol as symbol_mod
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown
    from optionsbot.strategies import Leg, StrategySuggestion

    scan_settings.scan.opening_range_fvg_enabled = True
    terminal_ev = -20.0
    suggestion = StrategySuggestion(
        strategy_name="long_call",
        legs=(
            Leg(
                symbol="SPY",
                side="buy",
                expiry=(date.today() + timedelta(days=45)).strftime("%Y%m%d"),
                strike=400.0,
                right="C",
            ),
        ),
        credit_or_debit=-100.0,
        max_loss=100.0,
        max_profit=None,
        prob_profit=0.60,
        suggested_quantity=1,
        defined_risk=True,
        rationale="test",
        expected_value=terminal_ev,
    )
    scored = ScoredStrategy(
        strategy_name="long_call",
        score=75.0,
        factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        suggestion=suggestion,
        rationale="test",
    )
    monkeypatch.setattr(symbol_mod, "score_all", MagicMock(return_value=(scored,)))
    raw_view = MarketView(
        direction="bear",
        direction_strength="weak",
        iv_regime="high",
        iv_rank_value=0.8,
        earnings_in_window=False,
        warming_up=False,
    )
    monkeypatch.setattr(symbol_mod, "infer_view", MagicMock(return_value=raw_view))
    start = datetime(2026, 8, 6, 13, 30, tzinfo=UTC)
    rows = [(99.5, 100.0, 99.0, 99.6, 100.0) for _ in range(10)]
    rows.extend(
        [
            (99.8, 100.7, 99.7, 100.5, 200.0),
            (100.5, 101.0, 100.45, 100.9, 150.0),
            (100.9, 101.1, 100.8, 101.0, 180.0),
            (100.7, 100.95, 100.7, 100.9, 250.0),
        ]
    )
    intraday = pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "volume"],
        index=[start + timedelta(minutes=i) for i in range(len(rows))],
    )
    signal = detect_opening_range_fvg(
        intraday,
        symbol="SPY",
        now=start + timedelta(minutes=15),
    )
    assert signal is not None and signal.quality is not None

    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,
        scan_engine,
        scan_settings,
        opening_range_signal=signal,
    )

    # Terminal PoP is not a target-before-stop probability. Until a calibrated
    # managed-path model is promoted, the exact 0DTE candidate is shadow-only.
    assert result.scored[0].suggestion.expected_value is None
    with scan_engine.connect() as conn:
        stored = conn.execute(
            select(strategy_scores).where(
                strategy_scores.c.snapshot_id == result.snapshot_id
            )
        ).one()
        snapshot = conn.execute(
            select(snapshots).where(snapshots.c.id == result.snapshot_id)
        ).one()
    assert stored.suggestion_json["expected_value"] is None
    assert stored.suggestion_json["gross_managed_expected_value"] is None
    assert stored.suggestion_json["managed_target_hit_probability_lcb"] is None
    assert stored.suggestion_json["managed_break_even_probability"] is not None
    assert stored.suggestion_json["estimated_round_trip_cost"] == pytest.approx(11.4)
    assert stored.suggestion_json["terminal_expected_value"] == terminal_ev
    assert stored.suggestion_json["expected_value_model"] == (
        "managed_outcome_calibration_required_v3"
    )
    quality = stored.suggestion_json["opening_range_fvg"]["quality"]
    assert quality["schema_version"] == "opening_range_quality_v1"
    assert quality["calibration_status"] == "shadow_unvalidated"
    assert quality["admission_enabled"] is False
    assert quality["regime"]["raw_direction"] == "bear"
    assert quality["regime"]["raw_iv_regime"] == "high"
    assert quality["regime"]["direction_opposed"] is True
    assert snapshot.raw_json["inferred_market_view"]["direction"] == "bear"
    assert snapshot.raw_json["configured_market_view"]["direction"] == "bear"
    assert snapshot.raw_json["effective_scoring_view"]["direction"] == "bull"
    assert snapshot.raw_json["effective_scoring_view"]["direction_strength"] == "strong"
    assert snapshot.regime_dir == "bull"
    assert snapshot.regime_iv == "neutral"


async def test_orb_scan_persists_optimizer_grid_only_as_shadow_scores(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    from optionsbot.analysis.opening_range_fvg import OpeningRangeFVGSignal
    from optionsbot.analysis.structure_optimizer import ShadowStructureCandidate
    from optionsbot.scan import symbol as symbol_mod
    from optionsbot.strategies import Leg

    scan_settings.scan.opening_range_fvg_enabled = True
    signal = OpeningRangeFVGSignal(
        signal_id="2026-08-28:SPY:bull:fvg-shadow-grid",
        session="2026-08-28",
        timeframe_minutes=1,
        direction="bull",
        opening_range_high=100.5,
        opening_range_low=99.5,
        breakout_ts=datetime(2026, 8, 28, 13, 41, tzinfo=UTC),
        fvg_formed_ts=datetime(2026, 8, 28, 13, 43, tzinfo=UTC),
        fvg_low=100.4,
        fvg_high=100.6,
        respected_ts=datetime(2026, 8, 28, 13, 45, tzinfo=UTC),
        entry_underlying_price=101.0,
        stop_pct=0.15,
        target_r=1.5,
        target_pct=0.225,
    )
    candidate = ShadowStructureCandidate(
        candidate_id="a" * 64,
        strategy="long_call_d50",
        legs=(
            Leg(
                symbol="SPY",
                side="buy",
                expiry="20260828",
                strike=101.0,
                right="C",
            ),
        ),
        entry_debit_dollars=100.0,
        maximum_loss_dollars=100.0,
        maximum_profit_dollars=20.0,
        round_trip_friction_dollars=11.4,
        desired_premium_target_dollars=22.5,
        premium_target_feasible=False,
        target_scenario_pnl_dollars=15.0,
        invalidation_scenario_pnl_dollars=-20.0,
        timeout_scenario_pnl_dollars=-8.0,
        features={
            "structure_kind": "long_option",
            "leg_count": 1,
            "friction_fraction": 0.114,
            "net_delta": 0.5,
            "net_gamma": 0.03,
            "net_theta": -0.12,
            "net_vega": 0.04,
            "thesis_entry_spot": 101.0,
            "thesis_invalidation_spot": 100.4,
            "thesis_target_spot": 101.9,
            "underlying_risk_fraction": 0.006,
            "underlying_reward_risk": 1.5,
            "timeout_minutes": 90.0,
            "premium_target_feasible": False,
        },
    )
    grid = MagicMock(return_value=(candidate,))
    monkeypatch.setattr(symbol_mod, "build_shadow_structure_grid", grid)
    monkeypatch.setattr(symbol_mod, "minutes_to_nyse_close", MagicMock(return_value=120.0))
    monkeypatch.setattr(symbol_mod, "score_all", MagicMock(return_value=()))

    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,
        scan_engine,
        scan_settings,
        opening_range_signal=signal,
    )

    # The alternative is persisted for managed capture but never enters the
    # ScanResult collection consumed by ranking, alerts, and execution.
    assert result.scored == ()
    with scan_engine.connect() as conn:
        rows = conn.execute(
            select(strategy_scores).where(
                strategy_scores.c.snapshot_id == result.snapshot_id
            )
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row.strategy == f"shadow_grid_v1:long_call_d50:{'a' * 64}"
    assert row.score == 0.0
    assert row.suggestion_json["shadow_only"] is True
    assert row.suggestion_json["admission_enabled"] is False
    assert row.suggestion_json["premium_target_feasible"] is False
    assert row.suggestion_json["managed_marketable_entry_net"] == -1.0
    assert row.suggestion_json["managed_marketable_basis_dollars"] == 100.0
    assert row.suggestion_json["managed_commission_estimate"] == pytest.approx(1.4)
    assert row.suggestion_json["structure_target_scenario_pnl_dollars"] == 15.0
    assert grid.call_args.kwargs["timeout_minutes"] == pytest.approx(90.0)


async def test_independent_hypothesis_persists_only_row_local_shadow_plan(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    import hashlib

    from optionsbot.analysis.intraday_hypotheses import (
        CausalWindow,
        MomentumMeasurements,
        OpeningMomentumFeatures,
        ShadowIntradayHypothesis,
    )
    from optionsbot.analysis.structure_optimizer import ShadowStructureCandidate
    from optionsbot.scan import symbol as symbol_mod
    from optionsbot.strategies import Leg

    scan_settings.scan.opening_range_fvg_enabled = True
    observed = datetime.now(UTC)
    signal_at = observed - timedelta(minutes=5)
    # The generator's thesis outlives the session safety boundary; persistence
    # must cap it at force-exit rather than keep the research lifetime.
    expires_at = observed + timedelta(minutes=120)
    features = OpeningMomentumFeatures(
        causal_window=CausalWindow(
            start_at=signal_at - timedelta(minutes=30),
            end_at=signal_at,
            last_bar_started_at=signal_at - timedelta(minutes=1),
            last_bar_completed_at=signal_at,
            bar_count=30,
        ),
        momentum=MomentumMeasurements(
            open_price=399.0,
            close_price=400.0,
            high_price=400.2,
            low_price=398.8,
            return_pct=1.0 / 399.0,
            directional_return_pct=1.0 / 399.0,
            range_pct=1.4 / 399.0,
            atr_14=0.8,
            directional_return_atr_ratio=1.25,
            directional_return_atr_normalized=1.25 / 2.25,
            directional_efficiency=0.7,
            directional_close_location=0.85,
            vwap=399.5,
            directional_vwap_distance_pct=0.5 / 399.5,
            vwap_direction_aligned=True,
            total_volume=3_000_000.0,
            mean_volume=100_000.0,
            relative_volume=1.4,
            relative_volume_normalized=1.4 / 2.4,
        ),
        opening_window_minutes=30,
        thesis_lifetime_minutes=90,
        second_half_volume_ratio=1.2,
        second_half_volume_normalized=1.2 / 2.2,
        parameter_version="intraday_shadow_windows_v1",
    )
    session = observed.date().isoformat()
    hypothesis = ShadowIntradayHypothesis(
        hypothesis_id="opening-momentum-causal-id",
        generator="opening_momentum_continuation",
        symbol="SPY",
        direction="bull",
        session=session,
        option_expiry=session.replace("-", ""),
        signal_at=signal_at,
        observed_at=observed,
        causal_cutoff_at=signal_at,
        thesis_expires_at=expires_at,
        reference_price=400.0,
        invalidation_level=399.0,
        features=features,
    )
    candidate = ShadowStructureCandidate(
        candidate_id="b" * 64,
        strategy="long_call_d50",
        legs=(
            Leg(
                symbol="SPY",
                side="buy",
                expiry=hypothesis.option_expiry,
                strike=400.0,
                right="C",
            ),
        ),
        entry_debit_dollars=100.0,
        maximum_loss_dollars=100.0,
        maximum_profit_dollars=None,
        round_trip_friction_dollars=11.4,
        desired_premium_target_dollars=22.5,
        premium_target_feasible=True,
        target_scenario_pnl_dollars=30.0,
        invalidation_scenario_pnl_dollars=-25.0,
        timeout_scenario_pnl_dollars=-8.0,
        features={
            "structure_kind": "long_option",
            "leg_count": 1,
            "friction_fraction": 0.114,
            "net_delta": 0.5,
            "net_gamma": 0.03,
            "net_theta": -0.12,
            "net_vega": 0.04,
            "thesis_entry_spot": 400.0,
            "thesis_invalidation_spot": 399.0,
            "thesis_target_spot": 401.5,
            "underlying_risk_fraction": 0.0025,
            "underlying_reward_risk": 1.5,
            "timeout_minutes": 60.0,
            "premium_target_feasible": True,
        },
    )
    grid = MagicMock(return_value=(candidate,))
    monkeypatch.setattr(symbol_mod, "build_shadow_grid_for_thesis", grid)
    monkeypatch.setattr(
        symbol_mod,
        "nyse_session_close_utc",
        MagicMock(return_value=observed + timedelta(minutes=90)),
    )
    monkeypatch.setattr(symbol_mod, "score_all", MagicMock(return_value=()))

    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,
        scan_engine,
        scan_settings,
        managed_hypotheses=(hypothesis,),
    )

    assert result.scored == ()
    with scan_engine.connect() as conn:
        snapshot = conn.execute(
            select(snapshots).where(snapshots.c.id == result.snapshot_id)
        ).one()
        row = conn.execute(
            select(strategy_scores).where(
                strategy_scores.c.snapshot_id == result.snapshot_id
            )
        ).one()
    plan = row.suggestion_json["managed_signal_plan"]
    assert plan["signal_id"] == hypothesis.hypothesis_id
    assert plan["generator"] == "opening_momentum_continuation"
    assert plan["admission_enabled"] is False
    assert plan["stop_pct"] == scan_settings.execution.opening_range_stop_pct
    assert plan["target_r"] == scan_settings.execution.opening_range_target_r_min
    persisted_expiry = datetime.fromisoformat(plan["thesis_expires_at"])
    assert abs(
        (persisted_expiry - (observed + timedelta(minutes=60))).total_seconds()
    ) < 2.0
    assert "opening_range_fvg" not in row.suggestion_json
    signal_hash = hashlib.sha256(hypothesis.hypothesis_id.encode()).hexdigest()[:12]
    assert row.strategy == (
        f"shadow_grid_v1:long_call_d50:{signal_hash}:{'b' * 64}"
    )
    assert snapshot.raw_json["managed_signal_plans"] == [plan]
    assert snapshot.raw_json["shadow_intraday_hypotheses"][0][
        "hypothesis_id"
    ] == hypothesis.hypothesis_id
    assert grid.call_args.kwargs["max_candidates"] <= 2


async def test_scan_symbol_passes_back_dte_gap_to_get_chain(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """scan_symbol fetches a back-month via settings.scan.back_month_dte_gap."""
    import optionsbot.scan.symbol as symbol_mod

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    kwargs = symbol_mod.ChainClient.return_value.get_chain.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["back_dte_gap"] == scan_settings.scan.back_month_dte_gap  # type: ignore[attr-defined]
