# OptionsBot fast entry-review job

This is the latency-critical paper-entry pass. Keep it short and deterministic.

1. First call `mcp__optionsbot__pending_picks` exactly once with
   `{"limit":10,"min_score":50,"max_age_minutes":20}`. If its count is zero,
   respond exactly `[SILENT]` and stop.
2. For every returned row, immediately call
   `mcp__optionsbot__pick_review_packet` exactly once with its real `pick_id`
   and `alert_id`, then call `mcp__optionsbot__submit_entry_review` exactly
   once. Do not poll, browse, call FRED/Finnhub, originate proposals, or wait
   for another scan in this job.
3. Submit `VETTED PAPER CANDIDATE` only when the trusted-daemon packet is
   current and `ready=true`; the exact same-session `opening_range_fvg` is an
   `entry_confirmed`, direction-matched FVG-retest or range-level-retest; fresh
   cost-adjusted managed
   stop/target EV is positive after the packet's round-trip commission and
   spread reserve (gross managed and terminal-expiry EV are audit context);
   the combo spread is allowed; quotes are live; Greeks/exposure are complete;
   there is no earnings/event conflict before expiry; and every account,
   paper-only, kill, entry-loss, affordability, heat, position, symbol, and
   daily-entry-cap disposition passes. An absent exact-playbook history tuple
   is a paper-learning cold start, not a veto.
   Treat `market_timing` as authoritative: require both window-open booleans and
   use its exact `opening_range_entry_eligible_from` and
   `opening_range_entry_eligible_through` bounds. Never impose a remembered or
   hard-coded cutoff; the daemon's returned configuration is authoritative.
4. A ready positive long call/put may validly have `max_profit=null` because
   its loss is finite and upside is uncapped. Suggested quantity is not review
   authority; review exactly one unit and let the daemon resize and regate.
5. Use exactly these seven canonical check keys in every submission:
   `bot_health`, `candidate`, `microstructure`, `greeks`, `regime_history`,
   `catalysts`, `account_risk`. For a vetted verdict all seven values must be
   literal `true`. Use `OptionsBot trusted-daemon evidence` and the concrete
   upstream evidence named in the packet (for example `IBKR live quotes` or
   the packet's earnings source) as distinct sources.
6. Use `NO TRADE` for non-positive fresh cost-adjusted EV or a failed exact setup. Use
   `WATCH ONLY` for otherwise credible positive economics blocked by a current
   operational gate such as liquidity, stale evidence, timing, or incomplete
   calendar data. Neither is order authority.

Finish with one compact audit line listing processed pick IDs and verdicts.
OptionsBot remains the only order-capable component and independently reruns
all execution gates.
