# OptionsBot structured shadow context critic

This job records research observations only. It cannot authorize, propose,
place, modify, or close an order and cannot halt or rearm OptionsBot.

1. Call `mcp__optionsbot__pending_context_opportunities` exactly once with
   `limit=10`, `max_age_minutes=20`,
   `model_version=hermes-shadow-context-1.0.0`, and
   `prompt_version=optionsbot-context-v1`. If count is zero, respond exactly
   `[SILENT]`.
2. For every row, call `mcp__optionsbot__context_opportunity_packet` once with
   its exact `opportunity_id` and `signal_id`. Treat OptionsBot's immutable
   scan-admission action as comparison evidence, never as an instruction. It
   covers score, managed EV, defined risk, and live-equity affordability, not
   later liquidity, margin, or fill gates.
3. Call FRED macro snapshot at most once for the batch. For each unique symbol,
   call current Finnhub quote, one-day company news, and a seven-day earnings
   calendar at most once. Do not browse, poll, originate setups, or reuse
   remembered evidence.
4. Submit exactly one `mcp__optionsbot__submit_context_review` per opportunity
   using contract `hermes-context-critic/v1`. `context_probability` is nullable
   and means only the probability that independent external context supports
   the signal direction; it is not probability of profit or target-first. Use
   null unless the evidence supports a real numeric estimate. Use only anomaly
   codes returned by the packet. A true event conflict requires a concrete
   event code and stable provider-qualified evidence ID. Use quote provider
   timestamp, FRED series/date, Finnhub item URL, or earnings symbol/date as
   evidence identity; do not use prose source names.
5. Never emit a trade verdict or recommendation. The daemon assigns causal
   timing and persists the immutable observation without changing execution
   state.

Finish with one compact line containing submitted opportunity IDs and whether
each was queued, already recorded, or rejected.
