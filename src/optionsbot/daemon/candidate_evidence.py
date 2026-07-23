"""Capture a bounded broker/account packet for one alerted candidate."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.economics import reconcile_entry_economics
from optionsbot.execution.equity_guard import new_entry_allowed
from optionsbot.execution.gate import can_execute
from optionsbot.execution.orders import open_orders
from optionsbot.execution.sizing import open_heat_dollars
from optionsbot.execution.state import load_state
from optionsbot.execution.walk import combo_bid_ask, combo_spread_issue
from optionsbot.ibkr import MarketDataClient, PositionsClient
from optionsbot.ibkr.types import OptionQuote
from optionsbot.storage.schema import orders, strategy_scores


def _number(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quote_dict(quote: OptionQuote, side: object, quantity: object) -> dict[str, Any]:
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


def apply_reconciled_economics(suggestion: object, evidence: dict[str, Any]) -> None:
    """Update an in-memory StrategySuggestion to match persisted fresh metrics."""
    economics = evidence.get("economics")
    if not isinstance(economics, dict):
        return
    for name in (
        "credit_or_debit",
        "max_loss",
        "max_profit",
        "reward_risk",
        "expected_value",
    ):
        if name in economics:
            setattr(suggestion, name, economics[name])


def _active_position_counts(context: DaemonContext, symbol: str) -> tuple[int, int]:
    with context.engine.connect() as conn:
        entries = conn.execute(
            select(orders.c.id, orders.c.symbol)
            .where(orders.c.intent == "open")
            .where(
                orders.c.status.in_(
                    ("staged", "submitting", "submitted", "partial", "filled")
                )
            )
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
    active = [row for row in entries if int(row.id) not in closed]
    return len(active), sum(1 for row in active if row.symbol == symbol)


async def capture_candidate_evidence(
    context: DaemonContext,
    *,
    score_id: int,
    symbol: str,
    legs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read evidence on the trusted daemon connection and persist it with the score."""
    issues: list[str] = []
    quote_map: dict[tuple[str, float, str], OptionQuote] = {}
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
                    symbol, spec[0], spec[1], spec[2]  # type: ignore[arg-type]
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    f"quote unavailable for {spec[0]} {spec[1]}{spec[2]}: "
                    f"{type(exc).__name__}"
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

    captured_at = datetime.now(UTC)

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
        with context.engine.connect() as conn:
            stored_suggestion = conn.execute(
                select(strategy_scores.c.suggestion_json).where(
                    strategy_scores.c.id == score_id
                )
            ).scalar_one()
        economics = reconcile_entry_economics(
            legs,
            dict(stored_suggestion or {}),
            fresh_net_per_share=mid,
        )
        if economics is None:
            issues.append("fresh executable economics unavailable")
        elif economics.expected_value is None:
            issues.append("fresh executable expected value unavailable")
        elif economics.expected_value <= 0:
            issues.append(
                "fresh executable expected value is non-positive "
                f"({economics.expected_value:.2f})"
            )

    state = load_state(context.engine)
    execution_gate = can_execute(context.settings, state)
    net_liq_usd = _number(summary.net_liquidation_usd) if summary is not None else None
    entry_gate = new_entry_allowed(
        context.engine, context.settings, current_net_liq=net_liq_usd
    )
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
    position_count_allowed = (
        active_count < context.settings.execution.max_open_positions
    )
    symbol_count_allowed = (
        symbol_count < context.settings.execution.max_per_symbol
    )
    if not single_trade_risk_allowed:
        issues.append("candidate exceeds or cannot prove the single-trade risk cap")
    if not portfolio_heat_allowed:
        issues.append("candidate exceeds or cannot prove the portfolio heat cap")
    if not position_count_allowed:
        issues.append("maximum open-position count reached")
    if not symbol_count_allowed:
        issues.append(f"maximum active {symbol} position count reached")
    evidence = {
        "schema_version": 1,
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
            "liquidity_issue": combo_issue,
        },
        "economics": economics.to_dict() if economics is not None else None,
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
