# OptionsBot independent intraday originator

This pass proposes paper hypotheses only; it does not review the alert queue.
Call `mcp__optionsbot__health` once. If the market or intraday entry window is
closed, respond `[SILENT]` and stop. The returned `market_timing` is
authoritative: use its exact eligible-from/through bounds and never impose a
remembered or hard-coded cutoff. Otherwise read one
current trusted snapshot each for SPY, QQQ, and IWM plus one current Finnhub
quote/news read per symbol. Propose at most one entry per symbol only when the
same-session snapshot contains a fresh `opening_range_fvg.status=entry_confirmed`
FVG-retest or range-level-retest setup matching the proposed direction,
confidence is at least 0.65, and checks
contains exactly `bot_health`, `regime_history`, and `catalysts`, all literal
`true`. Use two distinct named sources. Bull setups may use only long calls or
bull call spreads; bear setups may use only long puts or bear put spreads.

Never manufacture activity, waive a deterministic gate, call broker APIs, or
wait/poll for another scan. OptionsBot must independently rebuild and revalidate
every proposal before any paper order. End with one compact PROPOSED/SKIPPED
line for each symbol.
