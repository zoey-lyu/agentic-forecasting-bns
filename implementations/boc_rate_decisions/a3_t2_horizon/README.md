# A3 — The meeting after next (t+2 horizon)

> Plan step A3 of [`planning-docs/BOC_fable_enhancement_plan_260920.md`](../../../planning-docs/BOC_fable_enhancement_plan_260920.md),
> option 2. Self-contained folder like [`../a1_market_implied/`](../a1_market_implied/) and
> [`../a2_text_features/`](../a2_text_features/); it reuses both and modifies neither.

## The question

A1 found that nothing beats the futures curve four weeks before a decision. The
plan's reading was that this is a fact about *horizon*: four weeks out the next
decision is largely settled, but the one after it is ten weeks out and should be
priced less completely. A3 tests that directly.

Option 2 was chosen over the plan's own recommendation (statement tone shift)
for one reason: **A1's machinery extends to t+2, so the market can be scored at
this horizon too.** Tone shift has real labels neither in the data nor in the
market — it would be predicting our own extractor, with no reference to lose to.
Here the outcomes are real decisions and the market is still the opponent.

## Result 1 — the market's edge does decay with horizon

Same 13 target meetings at both leads, so the outcomes are identical and only
the lead time differs (RPS, lower is better):

| method | t+1 (4 weeks) | t+2 (~10 weeks) | skill vs market t+1 | skill vs market t+2 |
|---|---|---|---|---|
| market_implied | 0.113 | 0.155 | 0 | 0 |
| boc_logistic_macro_text | 0.147 | 0.170 | −0.30 | **−0.09** |
| boc_logistic_macro | 0.168 | 0.184 | −0.49 | **−0.18** |
| categorical_frequency | 0.214 | 0.218 | −0.90 | −0.41 |

The gap to the market closes by roughly two-thirds. **But read the columns, not
the skill: the gap narrows because the market gets worse (0.113 → 0.155), not
because our methods get better (they also degrade, just less).** Nobody here
gained anything at the longer horizon; the opponent weakened.

## Result 2 — the whole market-vs-model difference is about turning points

Splitting by what actually happened (descriptive — nobody can select the quiet
meetings in advance, so these are not bankable skill numbers):

**At t+2, 13 meetings — 3 moves, 10 holds**

| method | on the 3 moves | on the 10 holds |
|---|---|---|
| market_implied | **0.399** | 0.082 ← worst |
| boc_logistic_macro | 0.659 | 0.041 |
| boc_logistic_macro_text | 0.676 | **0.018** ← best |
| categorical_frequency | 0.853 | 0.028 |

**At t+1 (A1's window, 14 meetings — 4 cuts, 10 holds)**

| method | on the 4 cuts | on the 10 holds |
|---|---|---|
| market_implied | **0.205** | 0.069 ← worst |
| boc_logistic_macro | 0.505 | **0.025** |
| categorical_frequency | 0.837 | 0.029 |

The pattern is the same at both horizons and it reframes A1's headline. The
logistic is a *better* "nothing happens" predictor than the market — it beats it
by a wide margin on every hold. The market wins overall because it is far better
at the only meetings where a forecast is worth anything. It carries standing
insurance against a move, which costs it RPS on every quiet meeting and repays
on the rare live one.

So "the market wins" is really: **the market prices turns and our models do not.**

That is also why A2's text features helped exactly where they did. The two
meetings that carried A2's entire gain are the same easing-turn meetings that
carry the market's entire advantage here. Turn detection is the whole game, and
text is the only input so far that has shown any.

## Verdict

**The premise holds, weakly, and points somewhere specific.** There is more room
at t+2 than at t+1 — but it opened because the market's turn-detection decays
with horizon, not because anything of ours improved. The useful target is
therefore not "a longer horizon" but **"detect turns at a longer horizon"**,
where the market is measurably weaker (0.205 → 0.399 on moves) and text is the
only lead we have.

**Nothing here is statistically significant.** 13 meetings with 3 moves; every
paired interval covers zero (text vs market at t+2: +0.015, CI −0.069 to +0.110).
Treat all of it as direction, not as effect size.

Full numbers: [`data/leaderboard_a3.csv`](data/leaderboard_a3.csv),
[`data/horizon_comparison_a3.csv`](data/horizon_comparison_a3.csv),
[`data/outcome_decomposition_a3.csv`](data/outcome_decomposition_a3.csv),
[`data/per_meeting_rps_a3.csv`](data/per_meeting_rps_a3.csv).

## How the t+2 target is expressed

A task carries one fixed horizon, but the origin-to-target gap here varies
(63–77 days) because the Bank's calendar is irregular. Rather than bend the
harness, the **series** carries the shift:

- `timestamp` — the **t+1** meeting date, so `origin + 28 days` lands on it
  exactly as in the canonical task;
- `value` — the decision at the **next** meeting after that date;
- `released_at` — that later meeting's date, so cutoff enforcement hides the
  outcome until it is announced.

Every existing predictor works against it unchanged, because they all read
`task.target_series_id` generically. Climatology is run through a real
`ForecastContext` rather than A1's offline helper, because on this series the
outcome is published later than the row's own timestamp — filtering by
timestamp would leak.

## How the market is priced two meetings out

Two legs from the same curve:

1. `r1` — expected rate after the next meeting. A1's estimate, reused unchanged.
2. `r2` — expected rate after the meeting after that, from A1's two estimators
   with one change: inside the later meeting's month, the days *before* its
   decision sit at `r1`, not at today's CORRA, because the intervening decision
   has already happened by then.

`r2 − r1` is the implied move. Errors in `r1` propagate into `r2` — a real cost
of the horizon, reported rather than hidden: both legs are in every row's
`diagnostics`.

Feasibility was checked before building: across the 13 origins every t+2
contract is listed at the origin (the COA ladder is exactly four deep, so t+2
sits at its edge), seven origins get the clean next-month estimator and the
other seven a meeting-month split with 13–28 post-decision days.

## Known limits

- **13 meetings, 3 moves.** The move column — the one that matters — rests on
  three observations.
- **The outcome split conditions on the result.** It says where the difference
  lives, not what anyone could have banked.
- **No hikes**, as in every other BoC window here.
- **The two windows are not the same set of decisions.** The horizon comparison
  fixes this by restricting to shared target meetings; the standalone t+2
  leaderboard does not, so compare those numbers only within this table.
- **The text arm inherits A2's caveats**, including that the extraction is not
  cutoff-safe by construction.

## Files

| file | what it does |
|---|---|
| `data_t2.py` | Derives the shifted t+2 target series and registers it alongside the standard BoC series. |
| `implied_t2.py` | Two-leg futures decomposition: market-implied probabilities for the meeting after next. |
| `build_table_t2.py` | Prices every spec origin; writes `data/market_implied_t2.csv`. |
| `specs/boc_rate_direction_t2_eval.yaml` | The 13-origin t+2 spec. Exploratory, so it lives here rather than in `../specs/`. |
| `run_a3.py` | Scores the arms, writes the leaderboard, horizon comparison and outcome decomposition. |

Tests: `implementations/tests/boc_rate_decisions/test_a3_t2_horizon.py` (11 cases).

## Reproduce

```bash
cd implementations
python -m boc_rate_decisions.a3_t2_horizon.build_table_t2   # offline; reuses A1's fetched settlements
python -m boc_rate_decisions.a3_t2_horizon.run_a3           # no LLM calls
pytest tests/boc_rate_decisions/test_a3_t2_horizon.py
```
