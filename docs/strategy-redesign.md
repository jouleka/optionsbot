# 0DTE strategy and Hermes redesign

Status: implementation baseline, 2026-08-29

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

Use multiple independent, measurable hypotheses rather than treating FVG as a
universal premise:

1. Volatility-normalized opening momentum.
2. Failed breakout/re-entry mean reversion.
3. Late-session intraday momentum.
4. Explicit macro-event variants.
5. FVG geometry as a feature or challenger, not a mandatory truth.

Each generator must expire its thesis after bounded time, invalidation, or a
material return through the relevant level.

## Decision-time features

Persist the raw value and schema version for at least:

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

## Structure selection

Define the underlying thesis first: entry, invalidation, target, and expected
holding time. Scenario-price a bounded grid of liquid long options and verticals
at those states. Reject any finite structure whose desired target plus costs is
above its attainable payoff ceiling.

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

The daily selector chooses the best fresh, non-duplicate, sufficiently
independent opportunities. It does not let the first three marginal candidates
consume all capacity.

## Hermes boundary

Hermes is a context critic and research challenger. It may:

- convert macro/news evidence into a strict versioned schema;
- flag event conflicts or operational anomalies;
- propose research hypotheses and candidates in shadow mode;
- explain decisions and perform EOD attribution;
- compare its shadow action with the persisted bot baseline.

Hermes may not:

- supply an uncalibrated probability as trade authority;
- directly edit production strategy or risk configuration;
- place orders or bypass deterministic gates;
- own the global halt/rearm switch;
- receive credit for a trade the bot independently accepted or rejected.

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

IBKR does not provide expired-option historical data or historical combo data,
so these paths must be captured prospectively or sourced from a specialized
historical options dataset.

## Validation and promotion

- Replay option NBBO events, using ask-side buys, bid-side exits, latency, and
  conservative ordering when target and stop are ambiguous.
- Split by complete trading day and use rolling walk-forward folds with purging
  and embargo for overlapping labels.
- Retain a final untouched period.
- Record every model/parameter trial and estimate backtest-overfitting risk.
- Evaluate calibration (Brier/log loss/reliability), net expectancy, confidence
  intervals, profit factor, drawdown, slippage, and regime/fold stability.
- Compare Hermes only on opportunities where its pre-trade shadow action differs
  from the persisted baseline.
- Promote a challenger only when the lower confidence bound of after-cost value
  is positive across sufficient distinct sessions and operational tests pass.
- Version every promotion and support deterministic rollback.

## Implementation order

1. Correct economics, target feasibility, final costs, drift handling, settled
   position state, and halt authority. (Implemented in the 2026-08-29 baseline.)
2. Capture opportunity-level executable option paths prospectively.
3. Build the event-driven managed-outcome replay and calibration report.
4. Add the independent hypothesis generators and thesis-based structure grid as
   shadow challengers.
5. Add session-level portfolio ranking and underlying-aware invalidation exits.
6. Evaluate Hermes context features causally in shadow; promote only if they add
   out-of-sample incremental value.

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
