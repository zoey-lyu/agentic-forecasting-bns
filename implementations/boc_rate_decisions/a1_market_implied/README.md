# A1 — Market-implied probabilities as the BoC reference forecast

> Plan step A1 of [`planning-docs/BOC_fable_enhancement_plan_260920.md`](../../../planning-docs/BOC_fable_enhancement_plan_260920.md).
> Everything A1 adds lives in this folder; no existing module was modified.

## Why

Skill is `1 − RPS_model / RPS_reference`, so the reference defines what "good" means.
The BoC leaderboard scored skill against **climatology** — a baseline that reports the
historical hold frequency at every meeting and knows nothing else. Beating it proves
little. The **CORRA futures curve** is the real opponent: it already prices the press
releases, the data flow and the positioning of everyone trading Canadian rates.

A1 adds that reference and reports both skill columns, so every later experiment has a
single bar: **beating the market is what counts.**

## Result (protected eval window, RPS — lower is better)

| predictor | mean RPS | skill vs climatology | skill vs market | Δ vs market (95% CI) |
|---|---|---|---|---|
| **market_implied** | **0.108** | +0.58 | 0 | — |
| boc_logistic_macro | 0.162 | +0.38 | **−0.50** | +0.054 (−0.049, +0.177) |
| agent_anonymised | 0.170 | +0.35 | **−0.57** | +0.062 (−0.007, +0.136) |
| agent_identified | 0.224 | +0.14 | **−1.07** | +0.116 (+0.006, +0.223) |

The two agent rows are the **analyst agent** from the de-identification probe,
not the turn agent in [`../turn_detection/`](../turn_detection/). They are
different models on different tasks scored with different rules, and their
numbers are not comparable: this table is three-way RPS, that one is binary
Brier. `agent_anonymised` sees macro state with every date and the institution
stripped out, so recall is unavailable; `agent_identified` sees the real dates
and "Bank of Canada", so it is possible. Neither ever beats the market here.
| categorical_frequency | 0.260 | 0 | **−1.41** | +0.152 (−0.006, +0.334) |

All five rows cover the same 14 meetings. The two agent rows are **averaged over
two runs**, because the agent is not reproducible:

| arm | run 1 | run 2 | mean per-meeting spread |
|---|---|---|---|
| agent_identified | 0.240 | 0.208 | 0.032 |
| agent_anonymised | 0.160 | 0.179 | 0.022 |
| boc_logistic_macro | 0.1624 | 0.1624 | **0.000** |

Two runs minutes apart move 5 or 6 of the 14 cells and shift the mean by 0.02 to
0.03 — the same size as the differences the table is being read for. A single
agent run is one draw presented as a measurement, so every agent row here is the
average of the runs available; `run_leaderboard` reads every
`data/recall_probe_eval_run*/` directory it finds and says how many it used.
The logistic reproduces exactly, which is how we know the variation is the
model and not the harness.

This also reverses an earlier reading. On the first 12-meeting, single-run
version of this table `agent_anonymised` (0.161) came in *ahead* of the logistic
(0.186). With two more meetings and two runs averaged it falls behind (0.170
against 0.162). That ordering was a lucky draw.

Every method that looked skilful against climatology is unskilful against the market —
the logistic baseline that "won" the window loses ~48% more RPS than the futures curve.
**No gap is statistically significant**: twelve meetings cannot separate these methods,
and even climatology-vs-market has a CI that touches zero. Treat the ordering as the bar
to clear, not as a measured effect.

The market is not infallible here — it priced a 58% chance of a cut into the 2025-06-04
hold and only 32% into the 2025-09-17 cut — which is exactly what leaves room for a text
method to add something. A1 makes that room measurable.

Full numbers: [`data/leaderboard_a1.csv`](data/leaderboard_a1.csv),
per-meeting scores in [`data/per_meeting_rps_a1.csv`](data/per_meeting_rps_a1.csv).

## Where the probabilities come from

`COA` — **One-Month CORRA Futures** on the Montréal Exchange (TMX). A contract for
calendar month *M* settles at `100 − (average daily CORRA over M)`, so its settlement
price states the market's expected average overnight rate for that month. This is the
Canadian analogue of the Fed Funds futures behind CME FedWatch.

> **Correction to the 2026-08-24 data-source note**, which recorded CORRA futures as
> paywalled: `https://www.m-x.ca/en/trading/data/historical?symbol=COA&from=…&to=…&dnld=1`
> returns daily settlement CSV publicly, with no key and no registration. Responses cap at
> ~500 rows, so the fetch script chunks by month.
>
> History limits: `COA` (one-month) trades from January 2023 — enough for the whole eval
> window and for the 2023-24 replication window in [`../turn_detection/`](../turn_detection/)
> (the committed settlements cover 2023-01 onward), but not for the 2010–2024 backtest.
> `CRA` (three-month CORRA futures) goes back to
> 2020-06 and would need compounding arithmetic over a quarter. Neither reaches the
> backtest, which is moot: that window is already disqualified for LLM arms by recall
> contamination.

Two estimators, chosen by a rule fixed before looking at outcomes:

- **`next_month`** — when the month *after* the meeting holds no BoC announcement, that
  whole month sits at the post-decision rate, so `100 − settlement` *is* the expected
  post-decision rate. No day-count arithmetic at all.
- **`meeting_month`** — otherwise, split the meeting's own month at the decision's
  effective date and solve the month-average for the post-decision leg.

Rule: use `next_month` when it is available, else `meeting_month`. BoC meetings are about
six weeks apart, so a late-month decision — the case where `meeting_month` has only 1–3
days carrying the whole signal — always has a clean following month, and a mid-month
decision never does. Where both are computable and both are well conditioned the two
agree to about 1 bp (2025-03-19: −0.107 vs −0.095; 2026-06-17: +0.0097 vs +0.0100);
where `meeting_month` is thin it is visibly unreliable (2025-12-31: −0.362 on three days
vs −0.055), which is what the rule avoids. Both are stored in the `diagnostics` column of
every row.

Expected move → probabilities: one 25 bp step is a certain move, half a step is an even
chance, more than a step clamps to 1.0 (correct for a *direction* task that does not
distinguish 25 from 50 bp).

**Effective-date convention.** Since 2021 a decision takes effect the business day *after*
the announcement. Verified in the data: the 2025-01-29 cut moves CORRA on 2025-01-30
(3.29 → 3.03) and the target rate on the same day.

**Pipeline validation.** Reconstructing the realised month-average CORRA for all 23
expired contracts reproduces their final settlement price to a mean 0.5 bp. Re-running
the logistic baseline offline reproduces the stored harness per-meeting RPS exactly
(`max |ΔRPS| = 0.000000`).

## Known approximations

- CORRA sits a few bp above the target rate and spikes at month/quarter ends; that spread
  cancels in the expected *move* only if it is stable. Origins are read as a trailing
  5-business-day median of CORRA, which is robust to a single spike — two eval origins
  fall on 31 December.
- Open interest in COA is thin (0–800 contracts). Settlement prices are the exchange's
  official marks and are published for every listed contract, but a thin contract's mark
  can be stale. The two estimators agreeing is the available cross-check.
- All expectation is attributed to the single scheduled meeting. The 28-day lead
  guarantees no other scheduled announcement falls in between; an inter-meeting move
  would be misread.
- The agent rows are **stored** scores from the de-identification probe
  (`../data/recall_probe_eval/`), not fresh runs — re-running them costs LLM calls and
  spends the eval spec's `max_runs` budget for no new information. They also carry the
  known recall-contamination caveat, so read them as an upper bound.

## Files

| file | what it does |
|---|---|
| `fetch_market_data.py` | Downloads COA settlements (TMX) plus CORRA and the target rate (BoC Valet) into `data/`, 2023-01 onward by default. Retries transient exchange errors and coerces the placeholder expiry dates that appear in pre-2025 records. Network. |
| `implied.py` | Settlement price → `(p_cut, p_hold, p_hike)`. Pure functions. |
| `build_table.py` | Prices every origin in a spec; writes `data/market_implied_boc.csv`. |
| `market_implied.py` | `MarketImpliedPredictor` — replays that table as a harness `Predictor`. |
| `leaderboard.py` | Per-meeting RPS, offline climatology, two-reference skill table, paired bootstrap. |
| `run_leaderboard.py` | Assembles the leaderboard above from fresh and stored scores. |

Tests: `implementations/tests/boc_rate_decisions/test_a1_market_implied.py` (21 cases).

## Reproduce

```bash
cd implementations
python -m boc_rate_decisions.a1_market_implied.fetch_market_data   # network; --force to refresh
python -m boc_rate_decisions.a1_market_implied.build_table          # writes data/market_implied_boc.csv
python -m boc_rate_decisions.a1_market_implied.run_leaderboard      # writes data/leaderboard_a1.csv
pytest tests/boc_rate_decisions/test_a1_market_implied.py
```

`implementations/**/data/` is gitignored, so only the *derived* tables here are
committed (force-added): the implied-probability table, the leaderboard, and the
per-meeting scores. The raw fetched series — `coa_settlements.csv`,
`corra_daily.csv`, `target_rate_daily.csv` — are not, because
`fetch_market_data.py` regenerates them in about three minutes. So step 1 needs
network on a fresh clone; steps 2–4 do not.

`run_leaderboard` never calls an LLM and never touches the eval spec's run budget:
it scores the market table, recomputes climatology and the logistic baseline
directly, and reads the agent arms from disk.

## Deviations from the plan text

1. **Source.** The plan suggested scraping TMX's "Canadian Interest Rate Expectations"
   page, centralbank.watch or Wayback snapshots. The futures settlements are a better
   source — primary, free, exact, and not a secondary site's derived number — so all 14
   origins have a computed quote and none is hand-copied. No missing values to flag.
2. **Placement.** The plan puts the predictor at `predictors/market_implied.py` and edits
   `analysis.py::score_leaderboard` to take a reference argument. Both live here instead,
   because a large uncommitted backlog makes an isolated folder easier to review;
   `leaderboard.skill_table` covers the two-reference need without touching `analysis.py`.
   Moving them later is a file move plus one import line.
3. **Coverage.** The plan says 11 meetings. The eval spec was extended to 14 origins as
   part of this work (2026-09-02 resolved as a hold), and all 14 are priced. The
   leaderboard compares over the 12 that the stored agent runs cover.
