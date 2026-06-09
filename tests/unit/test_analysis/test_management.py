"""Tests for position-management triggers (IBK-113)."""

from __future__ import annotations

from datetime import date

from optionsbot.analysis.management import (
    evaluate_position_triggers,
    evaluate_profit_triggers,
)
from optionsbot.config import ManageSettings
from optionsbot.ibkr.types import PortfolioPosition

_TODAY = date(2026, 6, 8)


def _pp(
    strike: float = 95.0, right: str = "P", position: float = -1.0,
    expiry: str = "20260626", sec_type: str = "OPT",
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol="SPY", sec_type=sec_type,
        expiry=expiry if sec_type == "OPT" else None,
        strike=strike if sec_type == "OPT" else None,
        right=right if sec_type == "OPT" else None,  # type: ignore[arg-type]
        multiplier=100 if sec_type == "OPT" else 1, position=position, avg_cost=250.0,
        market_price=1.0, market_value=-100.0, unrealized_pnl=10.0, realized_pnl=0.0,
    )


def _settings() -> ManageSettings:
    return ManageSettings()  # manage_dte=21, urgent_dte=7


def test_dte_manage_bucket() -> None:
    out = evaluate_position_triggers([_pp(expiry="20260626")], {}, _TODAY, _settings())  # 18 DTE
    assert len(out) == 1 and out[0].triggers == ("dte_manage",) and out[0].dte == 18


def test_dte_urgent_bucket_only() -> None:
    out = evaluate_position_triggers([_pp(expiry="20260613")], {}, _TODAY, _settings())  # 5 DTE
    assert [a.triggers for a in out] == [("dte_urgent",)]


def test_no_alert_when_far_dated() -> None:
    assert evaluate_position_triggers([_pp(expiry="20260717")], {}, _TODAY, _settings()) == []


def test_assignment_short_put_itm() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260717")], {"SPY": 93.2}, _TODAY, _settings()
    )
    assert [a.triggers for a in out] == [("assignment",)]
    assert out[0].spot == 93.2 and out[0].itm is True
    assert out[0].dedup_key == "SPY:20260717:95:P:assignment"


def test_assignment_short_call_itm() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="C", expiry="20260717")], {"SPY": 97.0}, _TODAY, _settings()
    )
    assert [a.triggers for a in out] == [("assignment",)]


def test_otm_short_no_assignment() -> None:
    assert evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260717")], {"SPY": 99.0}, _TODAY, _settings()
    ) == []


def test_short_itm_near_expiry_merges_into_one_alert() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260613")], {"SPY": 92.0}, _TODAY, _settings()
    )  # 5 DTE + ITM put
    assert len(out) == 1
    assert out[0].triggers == ("assignment", "dte_urgent")  # sorted
    assert out[0].dedup_key == "SPY:20260613:95:P:assignment+dte_urgent"


def test_missing_spot_skips_assignment_keeps_dte() -> None:
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260613")], {}, _TODAY, _settings()
    )
    assert [a.triggers for a in out] == [("dte_urgent",)] and out[0].itm is None


def test_assignment_disabled_by_settings() -> None:
    s = ManageSettings(assignment_alerts=False)
    assert evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260717")], {"SPY": 90.0}, _TODAY, s
    ) == []


def test_stock_leg_ignored() -> None:
    assert evaluate_position_triggers(
        [_pp(sec_type="STK", position=-100.0)], {"SPY": 90.0}, _TODAY, _settings()
    ) == []


def test_expired_leg_no_dte_alert_but_assignment_fires() -> None:
    # Negative DTE (expiry already passed) must NOT fire a DTE alert -- otherwise it
    # would re-fire dte_urgent every cooldown. Assignment is independent and still fires.
    out = evaluate_position_triggers(
        [_pp(strike=95.0, right="P", expiry="20260601")], {"SPY": 90.0}, _TODAY, _settings()
    )
    assert [a.triggers for a in out] == [("assignment",)]  # dte = -7, no dte_* alert


def test_long_leg_near_expiry_itm() -> None:
    out = evaluate_position_triggers(
        [_pp(position=1.0, strike=95.0, right="P", expiry="20260613")], {"SPY": 92.0},
        _TODAY, _settings(),
    )  # long put, 5 DTE, ITM (spot < strike)
    assert len(out) == 1
    assert out[0].triggers == ("dte_urgent",)  # never assignment for a long
    assert out[0].itm is True


def test_long_leg_near_expiry_otm() -> None:
    out = evaluate_position_triggers(
        [_pp(position=1.0, strike=95.0, right="P", expiry="20260626")], {"SPY": 99.0},
        _TODAY, _settings(),
    )  # long put, 18 DTE, OTM
    assert out[0].triggers == ("dte_manage",) and out[0].itm is False


def test_long_leg_near_expiry_spot_unknown() -> None:
    out = evaluate_position_triggers(
        [_pp(position=1.0, strike=95.0, right="P", expiry="20260613")], {}, _TODAY, _settings()
    )
    assert out[0].triggers == ("dte_urgent",) and out[0].itm is None


def test_long_leg_far_dated_no_alert() -> None:
    assert evaluate_position_triggers(
        [_pp(position=1.0, expiry="20260717")], {"SPY": 90.0}, _TODAY, _settings()
    ) == []


def test_long_leg_alerts_disabled() -> None:
    s = ManageSettings(long_leg_expiry_alerts=False)
    assert evaluate_position_triggers(
        [_pp(position=1.0, strike=95.0, right="P", expiry="20260613")], {"SPY": 92.0}, _TODAY, s
    ) == []


# --- evaluate_profit_triggers (IBK-114) ------------------------------------


def _credit_leg(
    strike: float, right: str, position: float, avg_cost: float, upnl: float,
    expiry: str = "20260717",
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol="SPY", sec_type="OPT", expiry=expiry, strike=strike,
        right=right, multiplier=100, position=position, avg_cost=avg_cost,
        market_price=1.0, market_value=-100.0, unrealized_pnl=upnl, realized_pnl=0.0,
    )


def test_csp_take_profit_pins_avg_cost_scale() -> None:
    # CSP: $2.50 credit -> avg_cost 250, short 1. Up $125 -> 50% -> take_profit.
    out = evaluate_profit_triggers([_credit_leg(95.0, "P", -1.0, 250.0, 125.0)], ManageSettings())
    assert len(out) == 1
    a = out[0]
    assert a.trigger == "take_profit" and a.basis == "credit" and a.base_amount == 250.0
    assert a.net_pnl == 125.0
    assert round(a.profit_pct, 3) == 0.5 and a.dedup_key == "SPY:profit:take_profit"


def test_credit_spread_aggregates_legs() -> None:
    # short 250 + long -170 = 80 net credit; up 40 -> 50% -> take_profit.
    legs = [_credit_leg(95.0, "P", -1.0, 250.0, 60.0), _credit_leg(90.0, "P", 1.0, 170.0, -20.0)]
    out = evaluate_profit_triggers(legs, ManageSettings())
    assert len(out) == 1 and out[0].trigger == "take_profit"
    assert out[0].basis == "credit" and out[0].base_amount == 80.0 and out[0].net_pnl == 40.0


def test_stop_loss_fires_at_multiple() -> None:
    out = evaluate_profit_triggers([_credit_leg(95.0, "P", -1.0, 250.0, -500.0)], ManageSettings())
    assert [a.trigger for a in out] == ["stop_loss"]
    assert out[0].dedup_key == "SPY:profit:stop_loss"


def test_no_alert_between_thresholds() -> None:
    # +16% (below 50%), and loss not near -200% -> nothing.
    legs = [_credit_leg(95.0, "P", -1.0, 250.0, 40.0)]
    assert evaluate_profit_triggers(legs, ManageSettings()) == []


def test_min_credit_floor_skips_small() -> None:
    legs = [_credit_leg(95.0, "P", -1.0, 50.0, 40.0)]  # net_credit 50, would be 80%
    assert evaluate_profit_triggers(legs, ManageSettings(min_credit=100.0)) == []


def test_profit_alerts_disabled() -> None:
    legs = [_credit_leg(95.0, "P", -1.0, 250.0, 125.0)]
    assert evaluate_profit_triggers(legs, ManageSettings(profit_alerts=False)) == []


def test_custom_take_profit_threshold() -> None:
    legs = [_credit_leg(95.0, "P", -1.0, 250.0, 70.0)]  # 28%
    hits = evaluate_profit_triggers(legs, ManageSettings(take_profit_pct=0.25))
    assert hits[0].trigger == "take_profit"
    assert evaluate_profit_triggers(legs, ManageSettings(take_profit_pct=0.5)) == []


def test_default_min_credit_suppresses_near_zero() -> None:
    # A tiny / near-zero net credit (balanced or rolled book) must NOT alert under the
    # DEFAULT settings -- otherwise it produces a nonsense "X% of $0" alert (IBK-114 review).
    legs = [_credit_leg(95.0, "P", -1.0, 5.0, 4.0)]  # net_credit $5 < default floor (20)
    assert evaluate_profit_triggers(legs, ManageSettings()) == []


def test_multi_contract_scales_net_credit() -> None:
    # avg_cost is per-contract; |position| 2 -> net_credit 500. Up $250 -> 50% -> take_profit.
    out = evaluate_profit_triggers([_credit_leg(95.0, "P", -2.0, 250.0, 250.0)], ManageSettings())
    assert out[0].base_amount == 500.0 and out[0].trigger == "take_profit"


def test_stop_loss_boundary() -> None:
    s = ManageSettings()  # stop_loss_mult 2.0, net_credit 250 -> stop threshold -500
    at = [_credit_leg(95.0, "P", -1.0, 250.0, -500.0)]
    inside = [_credit_leg(95.0, "P", -1.0, 250.0, -499.0)]
    assert evaluate_profit_triggers(at, s)[0].trigger == "stop_loss"
    assert evaluate_profit_triggers(inside, s) == []  # just inside the threshold -> no alert


# --- debit branch (IBK-116) ------------------------------------------------


def _long_call(
    strike: float, avg_cost: float, upnl: float, position: float = 1.0,
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol="SPY", sec_type="OPT", expiry="20260717", strike=strike,
        right="C", multiplier=100, position=position, avg_cost=avg_cost,
        market_price=1.0, market_value=100.0, unrealized_pnl=upnl, realized_pnl=0.0,
    )


def test_long_call_debit_take_profit_pins_scale() -> None:
    # $2.50 debit -> avg_cost 250, long 1 -> net -250 -> debit base 250. Up $125 -> +50%.
    out = evaluate_profit_triggers([_long_call(100.0, 250.0, 125.0)], ManageSettings())
    assert len(out) == 1
    a = out[0]
    assert a.trigger == "take_profit" and a.basis == "debit" and a.base_amount == 250.0
    assert round(a.profit_pct, 3) == 0.5 and a.dedup_key == "SPY:profit:take_profit"


def test_long_call_debit_stop() -> None:
    out = evaluate_profit_triggers([_long_call(100.0, 250.0, -125.0)], ManageSettings())
    assert [a.trigger for a in out] == ["stop_loss"] and out[0].basis == "debit"


def test_debit_spread_aggregates_to_net_debit() -> None:
    # long 100C (avg 300, +1) + short 110C (avg 100, -1) -> net -300 + 100 = -200 debit.
    legs = [_long_call(100.0, 300.0, 60.0, position=1.0),
            _credit_leg(110.0, "C", -1.0, 100.0, 40.0)]
    out = evaluate_profit_triggers(legs, ManageSettings())
    assert len(out) == 1 and out[0].basis == "debit" and out[0].base_amount == 200.0
    assert out[0].trigger == "take_profit"  # net_pnl 100 >= 0.5 * 200


def test_min_debit_floor_skips_small() -> None:
    out = evaluate_profit_triggers([_long_call(100.0, 10.0, 8.0)], ManageSettings(min_debit=20.0))
    assert out == []


def test_balanced_zero_net_skipped() -> None:
    # short 100P (avg 100, -1) + long 100C (avg 100, +1) -> net 0 -> skipped.
    legs = [_credit_leg(95.0, "P", -1.0, 100.0, 50.0), _long_call(95.0, 100.0, 50.0)]
    assert evaluate_profit_triggers(legs, ManageSettings()) == []


def test_multi_contract_debit_scales() -> None:
    # avg_cost per-contract; long 2 -> debit base 500. Up $250 -> +50% -> take_profit.
    legs = [_long_call(100.0, 250.0, 250.0, position=2.0)]
    out = evaluate_profit_triggers(legs, ManageSettings())
    assert out[0].basis == "debit" and out[0].base_amount == 500.0
    assert out[0].trigger == "take_profit"


def test_debit_stop_boundary() -> None:
    s = ManageSettings()  # debit_stop_pct 0.5, debit base 250 -> stop threshold -125
    at = [_long_call(100.0, 250.0, -125.0)]
    inside = [_long_call(100.0, 250.0, -124.0)]
    assert evaluate_profit_triggers(at, s)[0].trigger == "stop_loss"
    assert evaluate_profit_triggers(inside, s) == []  # just inside -> no alert


# --- %-of-max-profit for defined-risk debit verticals (IBK-121) ------------


def test_debit_vertical_take_profit_sets_max_profit() -> None:
    # bull call: long 100C avg 300 (+1) + short 110C avg 100 (-1) -> net debit 200.
    # width 10 -> max_profit = 10*100 - 200 = 800. up 120 -> +60% on debit -> take_profit.
    legs = [_long_call(100.0, 300.0, 120.0, position=1.0),
            _credit_leg(110.0, "C", -1.0, 100.0, 0.0)]
    out = evaluate_profit_triggers(legs, ManageSettings())
    assert len(out) == 1 and out[0].trigger == "take_profit" and out[0].basis == "debit"
    assert out[0].max_profit == 800.0


def test_long_single_take_profit_no_max_profit() -> None:
    # a long call alone has unbounded max profit -> no %-of-max.
    out = evaluate_profit_triggers([_long_call(100.0, 250.0, 125.0)], ManageSettings())
    assert out[0].trigger == "take_profit" and out[0].max_profit is None


def test_credit_take_profit_no_max_profit() -> None:
    out = evaluate_profit_triggers([_credit_leg(95.0, "P", -1.0, 250.0, 125.0)], ManageSettings())
    assert out[0].basis == "credit" and out[0].max_profit is None


def test_debit_vertical_stop_no_max_profit() -> None:
    # stop only computes %-of-max for take-profits; a debit-vertical stop leaves it None.
    legs = [_long_call(100.0, 300.0, -200.0, position=1.0),
            _credit_leg(110.0, "C", -1.0, 100.0, 0.0)]
    out = evaluate_profit_triggers(legs, ManageSettings())
    assert out[0].trigger == "stop_loss" and out[0].max_profit is None


def test_debit_vertical_max_profit_nonpositive_omitted() -> None:
    # width 1 (100/101) but debit 150 -> max_profit = 100 - 150 = -50 <= 0 -> omitted.
    legs = [_long_call(100.0, 200.0, 120.0, position=1.0),
            _credit_leg(101.0, "C", -1.0, 50.0, 0.0)]  # net debit 150
    out = evaluate_profit_triggers(legs, ManageSettings())
    assert out[0].trigger == "take_profit" and out[0].max_profit is None
