# 0DTE strategy and Hermes redesign

Status: shadow-data implementation baseline, 2026-08-29. This document does
not claim deployment, a promoted model, or a demonstrated trading edge.

## Decision

The current opening-range/FVG rule is an experimental candidate generator, not
a demonstrated trading edge. Automatic admission must not use terminal-expiry
probability as a proxy for an intraday target-before-stop probability. Until a
managed option-path model is trained, calibrated, and promoted, exact-0DTE
candidates remain shadow observations with `expected_value=null`.

This does not alter the configured per-trade risk percentages. It fixes which
opportunities are allowed to consume that risk.

## Why the prior loop failed

The prior decision path was:

```text
opening-range/FVG pattern
  -> generic direction/IV score
  -> terminal-expiry option probability
  -> pretend that probability is target-first
  -> any EV > $0
  -> first candidates consume the daily slots
```

That path had several correctness errors:

- Expiry profitability is not the same event as reaching a premium target
  before a stop or timeout.
- The final execution reprice omitted the transaction-cost reserve.
- A finite debit spread could receive a target above its maximum attainable
  net profit.
- Material option-price drift reused stale probabilities and produced only a
  warning.
- An old breakout could claim an unrelated FVG formed nearly an hour later.
- The official backtest excludes DTE 0 and terminal outcome labels ignore the
  real stop/target/forced-exit path.
- Hermes reviews were non-causal and scored on the same unsuitable expiry
  labels; its free-form halt authority caused an operational false positive.

## Target decision pipeline

This is the intended end state, not a statement that every layer below is
implemented or production-authoritative today.

```text
complete intraday universe state
  -> independent hypothesis generators
  -> one stable opportunity_id per signal
  -> measurable feature vector at decision time
  -> liquid option-structure scenario grid
  -> calibrated managed-outcome probabilities
  -> conservative after-cost EV lower bound
  -> session-level rank/correlation selection
  -> deterministic risk + execution gates
  -> fill/exit/settlement and managed-path labels
  -> chronological challenger evaluation
```

No layer has a daily trade quota. A maximum number of entries is a ceiling, not
an objective.

## Hypothesis generators

The target set uses multiple independent, measurable hypotheses rather than
treating FVG as a universal premise:

1. Volatility-normalized opening momentum.
2. Failed breakout/re-entry mean reversion.
3. Late-session intraday momentum.
4. Explicit macro-event variants.
5. FVG geometry as a feature or challenger, not a mandatory truth.

Each generator must expire its thesis after bounded time, invalidation, or a
material return through the relevant level.

The current checkout implements the first three as deterministic shadow-only
generators. Explicit macro-event generators are not implemented. Their rows do
not enter `ScanResult`, alerts, execution, or base-model training; a separate
audited promotion bridge would be required before any could become an entry
candidate.

## Decision-time features

The target feature contract would persist the raw value and schema version for
at least:

- prior-close/open/session return;
- opening-range width divided by normal range for that exact time of day;
- breakout displacement, directional body/wick, gap size, retest depth, and
  rejection quality, normalized by intraday volatility;
- relative volume, VWAP distance/slope, market breadth, benchmark and sector
  confirmation;
- realized volatility versus remaining implied move, VIX1D/VIX, skew, and term
  structure where entitled;
- CPI, FOMC, payroll, GDP, earnings, and material issuer-event flags;
- minutes remaining, standardized moneyness, delta, gamma, theta, vega, and IV;
- NBBO width, displayed size, quote age, option volume, and open interest.

Do not overwrite the independently inferred market state with the setup
direction. Store both so disagreement is measurable.

The current causal allowlist implements a subset of this list. In particular,
it does not yet provide the complete breadth/sector, VIX/skew/term, macro-event,
displayed-size, quote-age, option-volume, or open-interest feature set. Missing
features must not be inferred or advertised as present.

## Structure selection

Define the underlying thesis first: entry, invalidation, target, and expected
holding time. The target system scenario-prices a bounded grid of liquid long
options and verticals at those states. Reject any finite structure whose desired
target plus costs is above its attainable payoff ceiling.

The implemented shadow grid is bounded and rejects missing bid/ask or Greeks,
but it is not itself a complete liquidity admission gate: it does not yet
enforce quote age, displayed size, option volume, open interest, or a configured
spread ceiling. Production execution retains separate live liquidity gates.
Its scenario values use a delta-gamma-theta approximation and are research
features, not a calibrated option repricer or evidence of edge.

For each structure `j`, estimate:

```text
EV_j = P(TP)_j * G_j
       - P(SL)_j * L_j
       + P(timeout)_j * E[timeout P&L]_j
       - C_j
```

`C_j` includes entry/exit commissions, bid/ask loss, modeled slippage, and fill
latency. `G_j`, `L_j`, and timeout P&L come from option prices, not an assumed
mapping from underlying direction.

For the simplified binary target/stop case:

```text
P_break_even = (L + C) / (G + L)
```

At a 15% stop and 1.5R target, frictionless break-even is 40%; costs raise it.
Admission uses a lower confidence bound, not a point estimate:

```text
edge_score = LCB_95(after_cost_EV) / maximum_capital_at_risk
```

The selector ranks each completed scan batch by conservative expected value per
dollar of defined risk, admits at most one structure per signal, and consumes
the remaining session capacity in chronological batch order. Once an earlier
opportunity consumes a daily slot, a later higher-ranked opportunity cannot
retroactively replace it; promotion evaluation replays that same causal rule.
The replay boundary is the deterministic, pre-dedup candidate selector. Live
alert cooldown/dedup may let a lower-ranked candidate fill an otherwise unused
delivery slot, so evaluation does not claim to reproduce Telegram delivery or
downstream fill side effects.

## Hermes boundary

Hermes is an asynchronous context critic and research challenger. Through the
restricted endpoint it may:

- convert macro/news evidence into a strict versioned schema;
- flag event conflicts or operational anomalies;
- explain decisions and perform EOD attribution;
- compare its shadow action with the persisted bot baseline.

Hermes may not:

- supply an uncalibrated probability as trade authority;
- directly edit production strategy or risk configuration;
- place orders or bypass deterministic gates;
- own the global halt/rearm switch;
- receive credit for a trade the bot independently accepted or rejected.

The restricted profile has no proposal or entry-review tool. Independent
hypotheses are generated deterministically by OptionsBot, not by Hermes. A
Hermes observation never changes the recorded bot action; the currently defined
shadow policy only measures whether an event-conflict hold would have avoided a
loss or missed a profit. Context artifacts are ineligible for live loading.

A future calibrated stack may combine independent bot and Hermes features:

```text
logit(P_final) = a
                 + b * logit(P_bot)
                 + c * logit(P_hermes_context)
                 + d' * regime_features
```

It must be fitted out of sample. Hand-written weighting is not a substitute.

## Required learning records

Add immutable, versioned records before training:

- `opportunities`: stable signal identity, session/symbol/setup, detection time,
  features and schema version;
- `candidate_decisions`: exact legs and quote, eligibility deadline, bot
  probability/action/EV, Hermes shadow output, final action, model/prompt/config
  versions and evidence hash;
- `candidate_marks`: timestamped executable option NBBO path for traded and
  shadow candidates;
- `managed_outcomes`: target/stop/timeout first-hit timestamps, entry/exit
  executable prices, MFE/MAE, spread, slippage, commissions and net P&L.

Persist the decision before an outcome can exist. Multiple structures and
repeated scans under one opportunity are correlated observations, not separate
trials.

### Implemented prospective capture contract

Migration `0023` implements the phase-one journal with these first-class
records:

- `managed_opportunities`: the first exact structure for a stable
  `(policy_version, signal_id, strategy)` key, frozen scan features, baseline
  capture action/reason, the later immutable OptionsBot scan-admission
  action/reason/timestamp, session close, entry cutoff, force-exit deadline,
  and the eventual managed label;
- `managed_opportunity_marks`: one idempotent executable synthetic-combo mark
  per opportunity and 15-second policy bucket, including unusable observations;
- `managed_context_reviews`: one immutable Hermes context response per
  opportunity and critic/prompt version, causally classified as pre-trade,
  post-entry, post-cutoff, or post-outcome;
- `managed_models` and `managed_model_evaluations`: immutable artifact and
  fold/holdout registries. Their presence does not load or promote a model.

Capture runs before score/EV, affordability, alert, and Hermes context
collection, so held candidates do not disappear from the learning population. Repeated scans
cannot change the first legs. Capacity admits one representative per independent
signal before alternative structures. Quote bundles rotate by signal and use
bounded concurrency and per-request deadlines under the daemon's short shared
lock budget, preventing one slow contract or early universe symbol from
starving later signals. Protective exits retain scheduling priority.

Three independent, deterministic generators now emit volatility-normalized
opening momentum, failed-breakout reversal, and late-session momentum theses
alongside the opening-range/FVG thesis. Each carries explicit decision-time
measurements, invalidation, target, expiry, and a stable signal identity. The
thesis-aware structure optimizer persists a bounded grid of exact-0DTE
long-option and debit-spread alternatives. Their ledger strategy identity is
`shadow_grid_v1:<structure>:<exact-leg-hash>`, while model features use the
low-cardinality structure portion. These rows freeze marketable entry basis,
commissions, spread friction, Greeks, target feasibility, and thesis scenarios,
but carry `shadow_only=true` and an immutable hold reason. They are deliberately
absent from `ScanResult`, ranking, alerts, execution, and the legacy terminal
outcome ledger. The managed-path journal is their only label source. Defense in
depth rejects a research-only row both when an execution request is loaded and
when an order is staged.

The base learner excludes these `shadow_only` alternatives today. Capturing a
managed-path label is not promotion authority; these generators remain
diagnostic until a future research-to-production gate is specified, tested, and
approved.

The default phase-one fill policy is deliberately conservative and versioned as
`marketable_nbbo_15s_v1`. Capture interval or polling phase, quote-age,
leg-synchrony, or maximum-mark-gap changes derive a different policy identity;
an explicitly configured stale identity fails closed. In the existing
signed-net convention:

```text
entry_net      = synthetic combo bid at detection
liquidation_net = synthetic combo ask at the later mark
gross P&L      = (entry_net - liquidation_net) * 100
net P&L        = gross P&L - estimated round-trip commissions
```

For a long debit this means buying at leg asks and liquidating at leg bids; for
a credit structure it means selling at bids and buying back at asks. The first
subsequent usable policy mark can resolve target or stop. The first usable mark
at/after the immutable force-exit deadline resolves timeout. Missing, delayed,
stale, partial, crossed, or materially asynchronous quotes never create a
label. A gap longer than the configured certainty window makes boundary order
ambiguous and resolves the opportunity as censored. The collector never writes
`managed_target_hit_probability`, changes execution state, or places an order.

IBKR does not provide expired-option historical data or historical combo data,
so these paths must be captured prospectively or sourced from a specialized
historical options dataset.

## Validation and promotion

- Replay option NBBO events, using ask-side buys, bid-side exits, latency, and
  conservative ordering when target and stop are ambiguous.
- Split by complete trading day and use rolling walk-forward folds with purging
  and embargo for overlapping labels.
- OOF eligibility freezes exactly one base challenger; it never promotes on the
  history used to create it, and no look-alike challenger may be minted while it
  waits.
- Evaluate the frozen artifact once on the shortest deterministic prefix of
  strictly later resolved sessions meeting the configured session and
  independent-signal minimums. That cohort is checksummed in an immutable
  `holdout` record.
- If an incumbent exists, score both frozen artifacts on the same untouched
  block and require positive incremental value before replacement. This serial
  one-challenger protocol prevents shopping the same future cohort across many
  trials.
- Evaluate calibration (Brier/log loss/reliability), net expectancy, confidence
  intervals, profit factor, drawdown, slippage, and regime/fold stability.
- Compare Hermes only on opportunities where its pre-trade shadow action differs
  from the persisted baseline.
- Promote a challenger only when both its walk-forward evidence and prospective
  after-cost mean-P&L lower confidence bound are positive, profit factor and
  coverage minimums pass, artifact/registry identities match, and all runtime
  gates remain paper-only.
- Version every promotion and support deterministic rollback.

The current collector observes policy marks on a configured polling cadence; it
is not tick-by-tick historical replay. It censors uncertain gaps, but a boundary
crossed and reversed entirely between polls may be unobserved. Unit and
integration tests validate these mechanics, not market profitability. A paper
RTH soak and prospective future-session evidence are still required.

## Implementation order

1. Correct economics, target feasibility, final costs, drift handling, settled
   position state, and halt authority. Implemented in the baseline.
2. Capture opportunity-level executable option paths prospectively. Implemented
   in source and migration `0023`; deployment and an RTH soak are separate.
3. Reduce polled managed paths and evaluate calibrated challengers. Implemented
   for prospective policy marks; tick/event historical replay is not.
4. Add independent hypotheses and a thesis-based structure grid. Three
   generators are implemented as shadow-only diagnostics; macro variants and a
   production promotion bridge are not.
5. Add session-level portfolio correlation selection and underlying-aware
   invalidation exits. Not implemented.
6. Evaluate Hermes context causally in shadow. The strict shadow ledger and
   evaluator are implemented in source; Hermes context cannot promote or alter
   admission, and deployment remains separate.

## Evidence references

- Options Industry Council, option price behavior:
  <https://www.optionseducation.org/referencelibrary/faq/option-price-behavior>
- Gao et al., intraday momentum:
  <https://profiles.wustl.edu/en/publications/market-intraday-momentum/>
- Federal Reserve, macro uncertainty in daily options:
  <https://www.federalreserve.gov/econres/ifdp/the-price-of-macroeconomic-uncertainty-evidence-from-daily-options.htm>
- Guo et al., probability calibration:
  <https://proceedings.mlr.press/v70/guo17a.html>
- IBKR historical-data limitations:
  <https://interactivebrokers.github.io/tws-api/historical_limitations.html>
- Federal Reserve model-risk guidance:
  <https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf>
