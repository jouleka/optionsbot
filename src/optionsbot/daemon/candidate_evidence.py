"""Capture a bounded broker/account packet for one alerted candidate."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update

from optionsbot.analysis.events import earnings_before_option_expiry
from optionsbot.analysis.positions import per_underlying_share_delta, portfolio_greeks
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import (
    is_market_open,
    minutes_to_nyse_close,
    nyse_session_date,
    nyse_session_start_utc,
)
from optionsbot.execution.economics import reconcile_entry_economics
from optionsbot.execution.equity_guard import new_entry_allowed
from optionsbot.execution.gate import can_execute
from optionsbot.execution.orders import open_orders
from optionsbot.execution.sizing import open_heat_dollars
from optionsbot.execution.state import load_state
from optionsbot.execution.walk import combo_bid_ask, combo_spread_issue
from optionsbot.ibkr import MarketDataClient, PositionsClient
from optionsbot.ibkr.types import OptionQuote
from optionsbot.storage.schema import (
    orders,
    position_settlements,
    snapshots,
    strategy_scores,
)
from optionsbot.strategies import StrategySuggestion


def _number(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quote_dict(quote: OptionQuote, side: object, quantity: object) -> dict[str, Any]:
    spread = (
        quote.ask - quote.bid
        if quote.bid is not None and quote.ask is not None and quote.ask >= quote.bid
        else None
    )
    spread_fraction = (
        spread / quote.mid
        if spread is not None and quote.mid is not None and quote.mid > 0
        else None
    )
    return {
        "symbol": quote.symbol,
        "expiry": quote.expiry,
        "strike": quote.strike,
        "right": quote.right,
        "side": side,
        "quantity": quantity,
        "bid": quote.bid,
        "ask": quote.ask,
        "mid": quote.mid,
        "spread": spread,
        "spread_fraction_of_mid": spread_fraction,
        "iv": quote.iv,
        "delta": quote.delta,
        "gamma": quote.gamma,
        "theta": quote.theta,
        "vega": quote.vega,
        "open_interest": quote.open_interest,
        "volume": quote.volume,
        "quote_ts": quote.ts.isoformat() if quote.ts is not None else None,
        "delayed": quote.delayed,
    }


def _candidate_greeks(
    legs: list[dict[str, Any]],
    quotes: dict[tuple[str, float, str], OptionQuote],
) -> dict[str, Any]:
    """Aggregate one strategy unit with explicit signs, multiplier, and units."""
    totals = {name: 0.0 for name in ("delta", "gamma", "theta", "vega")}
    option_legs = 0
    complete_legs = 0
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        option_legs += 1
        spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
        quote = quotes.get(spec)
        if quote is None:
            continue
        values = {name: getattr(quote, name) for name in totals}
        if any(value is None for value in values.values()):
            continue
        side = str(leg.get("side", "")).upper()
        if side not in {"BUY", "SELL"}:
            continue
        sign = 1.0 if side == "BUY" else -1.0
        quantity = float(leg.get("quantity", 1) or 1)
        scale = sign * quantity * 100.0
        for name, value in values.items():
            assert value is not None
            totals[name] += float(value) * scale
        complete_legs += 1
    complete = option_legs > 0 and complete_legs == option_legs
    return {
        "scope": "one_strategy_unit_all_option_legs",
        "option_multiplier": 100,
        "sign_convention": "BUY=+1, SELL=-1",
        "net_delta_share_equivalent": totals["delta"] if complete else None,
        "net_gamma_share_equivalent_per_dollar": totals["gamma"] if complete else None,
        "net_theta_dollars_per_day": totals["theta"] if complete else None,
        "net_vega_dollars_per_vol_point": totals["vega"] if complete else None,
        "option_legs_total": option_legs,
        "option_legs_with_complete_greeks": complete_legs,
        "complete": complete,
    }


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def with_reconciled_economics(
    suggestion: StrategySuggestion,
    evidence: dict[str, Any],
) -> StrategySuggestion:
    """Return a suggestion whose economics match the persisted fresh metrics."""
    economics = evidence.get("economics")
    if not isinstance(economics, dict):
        return suggestion
    updates = {
        name: economics[name]
        for name in (
            "credit_or_debit",
            "max_loss",
            "max_profit",
            "reward_risk",
            "expected_value",
        )
        if name in economics
    }
    if not updates:
        return suggestion
    return replace(suggestion, **updates)


def _active_position_counts(context: DaemonContext, symbol: str) -> tuple[int, int]:
    with context.engine.connect() as conn:
        entries = conn.execute(
            select(orders.c.id, orders.c.symbol)
            .where(orders.c.intent == "open")
            .where(orders.c.status.in_(("staged", "submitting", "submitted", "partial", "filled")))
        ).fetchall()
        closed = {
            int(row.closes_order_id)
            for row in conn.execute(
                select(orders.c.closes_order_id)
                .where(orders.c.intent == "close")
                .where(orders.c.status == "filled")
                .where(orders.c.closes_order_id.is_not(None))
            ).fetchall()
        }
        settled = {
            int(row.entry_order_id)
            for row in conn.execute(
                select(position_settlements.c.entry_order_id)
            ).fetchall()
        }
    active = [
        row
        for row in entries
        if int(row.id) not in closed and int(row.id) not in settled
    ]
    return len(active), sum(1 for row in active if row.symbol == symbol)


async def capture_candidate_evidence(
    context: DaemonContext,
    *,
    score_id: int,
    symbol: str,
    legs: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read evidence on the trusted daemon connection and persist it with the score."""
    issues: list[str] = []
    with context.engine.connect() as conn:
        stored = conn.execute(
            select(
                strategy_scores.c.suggestion_json,
                snapshots.c.raw_json,
                snapshots.c.spot,
            )
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(strategy_scores.c.id == score_id)
        ).one()
    stored_suggestion = dict(stored.suggestion_json or {})
    snapshot_raw = dict(stored.raw_json or {})
    candidate_spot = _number(stored.spot)

    quote_map: dict[tuple[str, float, str], OptionQuote] = {}
    portfolio_quote_map: dict[tuple[str, str, float, str], OptionQuote] = {}
    md = MarketDataClient(context.ibkr, resolver=context.resolver)
    positions_client = PositionsClient(context.ibkr)
    summary = None
    portfolio: list[Any] = []
    async with context.ibkr_lock:
        for leg in legs:
            if leg.get("sec_type", "OPT") != "OPT":
                continue
            spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
            try:
                quote_map[spec] = await md.get_option_review_snapshot(
                    symbol,
                    spec[0],
                    spec[1],
                    spec[2],  # type: ignore[arg-type]
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    f"quote unavailable for {spec[0]} {spec[1]}{spec[2]}: {type(exc).__name__}"
                )
        try:
            summary = await asyncio.wait_for(
                positions_client.get_account_summary(),
                timeout=context.settings.scan.scan_symbol_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"account summary unavailable: {type(exc).__name__}")
        try:
            portfolio = await asyncio.wait_for(
                positions_client.get_portfolio(),
                timeout=context.settings.scan.scan_symbol_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"portfolio unavailable: {type(exc).__name__}")

        # Existing option exposure needs the same explicit Greek scope as the
        # candidate. Reuse candidate quotes where identities overlap; fetch
        # only the remaining non-zero portfolio legs.
        for item in portfolio:
            if (
                item.sec_type != "OPT"
                or not item.position
                or item.expiry is None
                or item.strike is None
                or item.right is None
            ):
                continue
            key = (item.symbol, item.expiry, float(item.strike), str(item.right))
            candidate_key = (item.expiry, float(item.strike), str(item.right))
            if item.symbol == symbol and candidate_key in quote_map:
                portfolio_quote_map[key] = quote_map[candidate_key]
                continue
            try:
                portfolio_quote_map[key] = await md.get_option_snapshot(
                    item.symbol,
                    item.expiry,
                    float(item.strike),
                    item.right,
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    "portfolio Greek quote unavailable for "
                    f"{item.symbol} {item.expiry} {item.strike}{item.right}: "
                    f"{type(exc).__name__}"
                )

    captured_at = now if now is not None else datetime.now(UTC)

    quote_rows: list[dict[str, Any]] = []
    for leg in legs:
        if leg.get("sec_type", "OPT") != "OPT":
            continue
        spec = (str(leg["expiry"]), float(leg["strike"]), str(leg["right"]))
        quote = quote_map.get(spec)
        if quote is None:
            continue
        quote_rows.append(_quote_dict(quote, leg.get("side"), leg.get("quantity", 1)))
        label = f"{spec[0]} {spec[1]}{spec[2]}"
        if quote.delayed is not False:
            issues.append(f"delayed or unknown quote for {label}")
        if quote.ts is None:
            issues.append(f"missing quote timestamp for {label}")
        else:
            quote_ts = quote.ts if quote.ts.tzinfo is not None else quote.ts.replace(tzinfo=UTC)
            age = (captured_at - quote_ts.astimezone(UTC)).total_seconds()
            if age < -60 or age > context.settings.execution.entry_quote_max_age_seconds:
                issues.append(f"stale quote for {label}")
        if quote.bid is None or quote.ask is None or quote.mid is None or quote.ask < quote.bid:
            issues.append(f"invalid bid/ask for {label}")
        missing_greeks = [
            name
            for name in ("iv", "delta", "gamma", "theta", "vega")
            if getattr(quote, name) is None
        ]
        if missing_greeks:
            issues.append(f"missing Greeks for {label}: {','.join(missing_greeks)}")
        if quote.open_interest is None or quote.volume is None:
            issues.append(f"missing OI/volume for {label}")

    combo = combo_bid_ask(legs, quote_map)
    mid = (combo[0] + combo[1]) / 2 if combo is not None else None
    combo_issue = None
    economics = None
    if mid is None:
        issues.append("combo mid unavailable")
    else:
        combo_issue = combo_spread_issue(
            legs,
            quote_map,
            mid,
            max_frac=context.settings.execution.max_combo_spread_frac,
        )
        if combo_issue is not None:
            issues.append("liquidity: " + combo_issue)
        economics = reconcile_entry_economics(
            legs,
            stored_suggestion,
            fresh_net_per_share=mid,
        )
        if economics is None:
            issues.append("fresh executable economics unavailable")
        elif economics.expected_value is None:
            issues.append("fresh executable expected value unavailable")
        elif economics.expected_value <= 0:
            issues.append(
                f"fresh executable expected value is non-positive ({economics.expected_value:.2f})"
            )

    state = load_state(context.engine)
    execution_gate = can_execute(context.settings, state)
    net_liq_usd = _number(summary.net_liquidation_usd) if summary is not None else None
    entry_gate = new_entry_allowed(context.engine, context.settings, current_net_liq=net_liq_usd)
    if not execution_gate.allowed:
        issues.append("execution interlock: " + execution_gate.reason)
    if not entry_gate.allowed:
        issues.append("entry loss guard: " + entry_gate.reason)
    account = {
        "currency": summary.currency if summary is not None else None,
        "net_liquidation": _number(summary.net_liquidation) if summary is not None else None,
        "net_liquidation_usd": net_liq_usd,
        "buying_power": _number(summary.buying_power) if summary is not None else None,
        "available_funds": _number(summary.available_funds) if summary is not None else None,
    }
    for name in ("net_liquidation_usd", "buying_power", "available_funds"):
        if account[name] is None:
            issues.append(f"account {name} unavailable")

    active_count, symbol_count = _active_position_counts(context, symbol)
    try:
        open_heat = open_heat_dollars(context.engine)
    except Exception as exc:  # noqa: BLE001
        open_heat = None
        issues.append(f"portfolio heat unavailable: {type(exc).__name__}")
    candidate_max_loss = economics.max_loss if economics is not None else None
    single_trade_cap = (
        net_liq_usd * context.settings.execution.max_single_trade_risk_pct
        if net_liq_usd is not None
        else None
    )
    portfolio_heat_cap = (
        net_liq_usd * context.settings.execution.max_portfolio_heat_pct
        if net_liq_usd is not None
        else None
    )
    single_trade_risk_allowed = (
        candidate_max_loss <= single_trade_cap
        if candidate_max_loss is not None and single_trade_cap is not None
        else False
    )
    portfolio_heat_allowed = (
        open_heat + candidate_max_loss <= portfolio_heat_cap
        if open_heat is not None
        and candidate_max_loss is not None
        and portfolio_heat_cap is not None
        else False
    )
    position_count_allowed = active_count < context.settings.execution.max_open_positions
    symbol_count_allowed = symbol_count < context.settings.execution.max_per_symbol
    candidate_bp_reserve = candidate_max_loss
    candidate_bp_usage_pct = (
        candidate_bp_reserve / net_liq_usd
        if candidate_bp_reserve is not None and net_liq_usd is not None and net_liq_usd > 0
        else None
    )
    available_funds = _number(summary.available_funds) if summary is not None else None
    candidate_affordable = (
        candidate_bp_reserve <= available_funds
        if candidate_bp_reserve is not None and available_funds is not None
        else False
    )
    deployment_after = (
        open_heat + candidate_max_loss
        if open_heat is not None and candidate_max_loss is not None
        else None
    )
    deployment_after_pct = (
        deployment_after / net_liq_usd
        if deployment_after is not None and net_liq_usd is not None and net_liq_usd > 0
        else None
    )
    bp_deployment_allowed = (
        deployment_after_pct <= context.settings.execution.max_bp_usage_pct
        if deployment_after_pct is not None
        else False
    )
    if not single_trade_risk_allowed:
        issues.append("candidate exceeds or cannot prove the single-trade risk cap")
    if not portfolio_heat_allowed:
        issues.append("candidate exceeds or cannot prove the portfolio heat cap")
    if not position_count_allowed:
        issues.append("maximum open-position count reached")
    if not symbol_count_allowed:
        issues.append(f"maximum active {symbol} position count reached")
    if not candidate_affordable:
        issues.append("candidate structural BP reserve exceeds or cannot prove available funds")
    if not bp_deployment_allowed:
        issues.append("candidate exceeds or cannot prove the OptionsBot BP deployment cap")

    candidate_greeks = _candidate_greeks(legs, quote_map)
    if candidate_greeks["complete"] is not True:
        issues.append("candidate aggregate Greeks incomplete")
    active_portfolio = [item for item in portfolio if item.position]
    existing_greeks = portfolio_greeks(active_portfolio, portfolio_quote_map)
    if existing_greeks["complete"] is not True:
        issues.append("existing portfolio aggregate Greeks incomplete")
    existing_delta_by_symbol = per_underlying_share_delta(
        active_portfolio,
        portfolio_quote_map,
    )
    candidate_share_delta = candidate_greeks["net_delta_share_equivalent"]
    beta_value = _number(snapshot_raw.get("beta_to_benchmark"))
    candidate_dollar_delta = (
        candidate_share_delta * candidate_spot
        if isinstance(candidate_share_delta, float) and candidate_spot is not None
        else None
    )
    incremental_beta_weighted_dollar_delta = (
        candidate_dollar_delta * beta_value
        if candidate_dollar_delta is not None and beta_value is not None
        else None
    )
    incremental_beta_weighted_delta_pct_of_net_liq = (
        incremental_beta_weighted_dollar_delta / net_liq_usd
        if incremental_beta_weighted_dollar_delta is not None
        and net_liq_usd is not None
        and net_liq_usd > 0
        else None
    )
    beta_delta_complete = (
        candidate_share_delta is not None
        and candidate_spot is not None
        and beta_value is not None
    )
    if not beta_delta_complete:
        issues.append("candidate incremental beta/delta impact incomplete")

    now_session = nyse_session_date(captured_at)
    minutes_to_close = minutes_to_nyse_close(captured_at)
    market_open = is_market_open(captured_at)
    entry_window_open = market_open and (
        not context.settings.execution.zero_dte_only
        or (
            minutes_to_close is not None
            and minutes_to_close
            > context.settings.execution.zero_dte_entry_cutoff_minutes
        )
    )
    opening_range_window_open = True
    if context.settings.scan.opening_range_fvg_enabled:
        market_open_at = nyse_session_start_utc(captured_at) + timedelta(
            hours=9, minutes=30
        )
        opening_range_window_open = (
            market_open_at
            + timedelta(minutes=context.settings.scan.opening_range_minutes)
            <= captured_at
            <= market_open_at
            + timedelta(
                minutes=context.settings.scan.opening_range_entry_window_minutes
            )
        )
        entry_window_open = entry_window_open and opening_range_window_open
    if not entry_window_open:
        issues.append("market or configured strategy entry window is closed")

    option_expiries = sorted(
        {
            str(leg["expiry"])
            for leg in legs
            if leg.get("sec_type", "OPT") == "OPT" and leg.get("expiry")
        }
    )
    next_earnings = _parse_date(snapshot_raw.get("next_earnings_date"))
    earnings_before_expiry = earnings_before_option_expiry(
        next_earnings,
        option_expiries,
        today=now_session,
    )
    short_option_legs = sum(
        1
        for leg in legs
        if leg.get("sec_type", "OPT") == "OPT"
        and str(leg.get("side", "")).upper() == "SELL"
    )
    physically_settled = symbol.upper() not in {"SPX", "SPXW", "XSP"}
    expires_this_session = option_expiries == [now_session.strftime("%Y%m%d")]

    combo_spread = combo[1] - combo[0] if combo is not None else None
    combo_spread_fraction = (
        combo_spread / abs(mid)
        if combo_spread is not None and mid is not None and mid != 0
        else None
    )
    incident = getattr(context.ibkr, "competing_live_session_status", None)
    if not isinstance(incident, dict):
        incident = {
            "active_recently": False,
            "last_observed_at": None,
            "count_since_connect": 0,
            "error_code": 10197,
        }
    evidence = {
        "schema_version": 2,
        "source": "trusted_daemon",
        "score_id": score_id,
        "captured_at": captured_at.isoformat(),
        "ready": not issues,
        "readiness_issues": issues,
        "option_quotes": quote_rows,
        "combo": {
            "bid": combo[0] if combo is not None else None,
            "ask": combo[1] if combo is not None else None,
            "mid": mid,
            "spread": combo_spread,
            "spread_fraction_of_net_premium": combo_spread_fraction,
            "max_spread_fraction": context.settings.execution.max_combo_spread_frac,
            "spread_allowed": combo_issue is None and combo is not None,
            "liquidity_issue": combo_issue,
        },
        "economics": economics.to_dict() if economics is not None else None,
        "candidate_scope": {
            "strategy_units_reviewed": 1,
            "economics_scope": "one_strategy_unit",
            "risk_scope": "one_strategy_unit",
            "greeks_scope": "one_strategy_unit",
            "execution_quantity_recomputed_by_daemon": True,
        },
        "candidate_greeks": candidate_greeks,
        "exposure": {
            "beta_benchmark": snapshot_raw.get("beta_benchmark"),
            "candidate_spot": candidate_spot,
            "candidate_beta_to_benchmark": beta_value,
            "candidate_share_delta": candidate_share_delta,
            "candidate_dollar_delta": candidate_dollar_delta,
            "incremental_beta_weighted_dollar_delta": (
                incremental_beta_weighted_dollar_delta
            ),
            "incremental_beta_weighted_delta_pct_of_net_liq": (
                incremental_beta_weighted_delta_pct_of_net_liq
            ),
            "beta_delta_hard_cap_configured": False,
            "beta_delta_policy": (
                "measured analyst input; daemon has no separate beta/delta hard cap"
            ),
            "existing_portfolio_greeks": existing_greeks,
            "existing_share_delta_by_symbol": existing_delta_by_symbol,
            "existing_candidate_symbol_share_delta": existing_delta_by_symbol.get(
                symbol, 0.0
            ),
            "after_candidate_symbol_share_delta": (
                existing_delta_by_symbol.get(symbol, 0.0) + candidate_share_delta
                if isinstance(candidate_share_delta, float)
                else None
            ),
            "complete": beta_delta_complete and existing_greeks["complete"] is True,
        },
        "expiration_assignment": {
            "option_expiries": option_expiries,
            "expires_this_session": expires_this_session,
            "short_option_legs": short_option_legs,
            "physically_settled": physically_settled,
            "early_assignment_possible": physically_settled and short_option_legs > 0,
            "pin_risk_present": expires_this_session and short_option_legs > 0,
            "force_exit_minutes_before_close": (
                context.settings.execution.zero_dte_force_exit_minutes
            ),
            "handling": (
                "daemon force-exit and expiration settlement/assignment reconciliation "
                "remain authoritative"
            ),
        },
        "catalysts": {
            "analysis_earnings_window": snapshot_raw.get("earnings_in_window"),
            "next_earnings_date": snapshot_raw.get("next_earnings_date"),
            "earnings_source": snapshot_raw.get("earnings_source"),
            "issuer_earnings_applicable": (
                snapshot_raw.get("earnings_source") != "not_applicable"
            ),
            "earnings_before_option_expiry": earnings_before_expiry,
        },
        "market_timing": {
            "captured_at": captured_at.isoformat(),
            "nyse_session": now_session.isoformat(),
            "market_open": market_open,
            "minutes_to_close": minutes_to_close,
            "zero_dte_only": context.settings.execution.zero_dte_only,
            "zero_dte_entry_cutoff_minutes": (
                context.settings.execution.zero_dte_entry_cutoff_minutes
            ),
            "entry_window_open": entry_window_open,
            "opening_range_window_open": opening_range_window_open,
        },
        "opening_range_fvg": snapshot_raw.get("opening_range_fvg"),
        "market_data_incident": incident,
        "account": account,
        "positions": [
            {
                "symbol": item.symbol,
                "sec_type": item.sec_type,
                "expiry": item.expiry,
                "strike": item.strike,
                "right": item.right,
                "position": item.position,
                "market_value": item.market_value,
                "unrealized_pnl": item.unrealized_pnl,
            }
            for item in portfolio
        ],
        "risk": {
            "execution_allowed": execution_gate.allowed,
            "execution_reason": execution_gate.reason,
            "entry_loss_guard_allowed": entry_gate.allowed,
            "entry_loss_guard_reason": entry_gate.reason,
            "paper_only": context.settings.execution.paper_only,
            "paper_account": context.settings.ibkr.paper,
            "port": context.settings.ibkr.port,
            "mode": context.settings.execution.mode,
            "max_open_positions": context.settings.execution.max_open_positions,
            "max_per_symbol": context.settings.execution.max_per_symbol,
            "active_positions": active_count,
            "active_symbol_positions": symbol_count,
            "open_orders": len(open_orders(context.engine)),
            "open_heat": open_heat,
            "max_portfolio_heat_pct": context.settings.execution.max_portfolio_heat_pct,
            "max_single_trade_risk_pct": context.settings.execution.max_single_trade_risk_pct,
            "max_bp_usage_pct": context.settings.execution.max_bp_usage_pct,
            "candidate_bp_reserve_method": "structural_max_loss_pre_review",
            "candidate_bp_reserve": candidate_bp_reserve,
            "candidate_bp_usage_pct_of_net_liq": candidate_bp_usage_pct,
            "available_funds_after_candidate_reserve": (
                available_funds - candidate_bp_reserve
                if available_funds is not None and candidate_bp_reserve is not None
                else None
            ),
            "candidate_affordable": candidate_affordable,
            "optionsbot_deployment_after": deployment_after,
            "optionsbot_deployment_after_pct": deployment_after_pct,
            "bp_deployment_allowed": bp_deployment_allowed,
            "broker_whatif_required_at_execution": True,
            "candidate_max_loss": candidate_max_loss,
            "single_trade_cap": single_trade_cap,
            "single_trade_risk_allowed": single_trade_risk_allowed,
            "portfolio_heat_after": (
                open_heat + candidate_max_loss
                if open_heat is not None and candidate_max_loss is not None
                else None
            ),
            "portfolio_heat_cap": portfolio_heat_cap,
            "portfolio_heat_allowed": portfolio_heat_allowed,
            "position_count_allowed": position_count_allowed,
            "symbol_count_allowed": symbol_count_allowed,
            "last_reconcile_at": (
                context.last_reconcile_ts.isoformat()
                if context.last_reconcile_ts is not None
                else None
            ),
        },
    }
    with context.engine.begin() as conn:
        suggestion = conn.execute(
            select(strategy_scores.c.suggestion_json).where(strategy_scores.c.id == score_id)
        ).scalar_one()
        updated = dict(suggestion or {})
        if economics is not None:
            updated.update(
                credit_or_debit=economics.credit_or_debit,
                max_loss=economics.max_loss,
                max_profit=economics.max_profit,
                reward_risk=economics.reward_risk,
                expected_value=economics.expected_value,
            )
        updated["review_evidence"] = evidence
        conn.execute(
            update(strategy_scores)
            .where(strategy_scores.c.id == score_id)
            .values(suggestion_json=updated)
        )
    return evidence
