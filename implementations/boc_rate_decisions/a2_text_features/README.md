# A2 — Statement text as predictor features (extract, don't predict)

> Plan step A2 of [`planning-docs/BOC_fable_enhancement_plan_260920.md`](../../../planning-docs/BOC_fable_enhancement_plan_260920.md).
> Self-contained folder, like [`../a1_market_implied/`](../a1_market_implied/); it reuses A1's
> scoring helpers and adds nothing to existing modules.

## The idea

Split the job along the line the evidence draws. Asking an LLM for a probability
loses to a four-feature logistic on this window, and the edge it once appeared to
have was recall of the answer. Asking an LLM to read a statement and score its
tone is a different task: bounded output, low variance, and checkable against the
sentence it quotes.

So the model never sees a forecasting question here. It returns a structured
record per statement — `hawk_dove` (−2..+2 toward the path ahead), `guidance`
(five ordered levels), cited `drivers`, and a verbatim `evidence` quote — and a
logistic regression fit at the origin decides what that record is worth.

## Result (protected eval window, 14 meetings, RPS — lower is better)

| arm | mean RPS | skill vs climatology | skill vs market | Δ vs macro-only (95% CI) |
|---|---|---|---|---|
| market_implied (A1 reference) | **0.108** | +0.58 | 0 | — |
| **boc_logistic_macro_text** | **0.145** | +0.44 | −0.35 | **−0.017** (−0.091, +0.073) |
| boc_logistic_macro (ablation) | 0.162 | +0.38 | −0.50 | 0 |
| boc_logistic_macro_text_shuffled | 0.186 | +0.28 | −0.72 | +0.023 (−0.015, +0.077) |
| categorical_frequency | 0.260 | 0 | −1.41 | +0.098 |

Text features cut the gap to the market by about a third (−0.50 → −0.35 skill),
and the ordering — real text better than macro-only better than shuffled text —
is what a genuine signal looks like. **But the paired interval on 14 meetings
covers zero, and the gain is not spread across the window.**

### Where the gain actually comes from

| | text | macro-only | Δ |
|---|---|---|---|
| all 14 meetings | 0.145 | 0.162 | **−0.017** |
| excluding 2025-09-17 and 2025-10-29 | 0.072 | 0.049 | **+0.023** |

Two meetings carry the entire result. They are the September and October 2025
cuts — the easing turn, where the macro baseline scored 0.91 and 0.78 (its two
worst meetings of the window) and the stance features had flagged the preceding
statements as dovish (`hawk_dove = −1`, `guidance = conditional_easing`). Take
those two away and the text arm is *worse* than macro-only everywhere else, and
worst of all at 2025-03-12 (0.68 vs 0.23).

The honest reading: **the statement text helps at a turn and costs a little the
rest of the time.** That is the same shape as the search-vs-no-search finding of
2026-08-24 — a mechanism that fixes regime turns, not a uniform improvement —
and it should not be extrapolated to windows without a turn in them.

Full numbers: [`data/leaderboard_a2.csv`](data/leaderboard_a2.csv),
[`data/per_meeting_rps_a2.csv`](data/per_meeting_rps_a2.csv).

## Controls

**Ablation.** The macro-only arm is the same class with the feature block
switched off, and the run script checks it reproduces A1's stored per-meeting
RPS exactly (`max |ΔRPS| = 0`), aborting if it does not. Without that check the
comparison would not be a text comparison.

**Shuffled stances.** Same three features, same parameter count, same fit — but
each meeting is handed another meeting's reading. It scores *worse* than using no
text at all, which is the expected sign: three noise columns spend degrees of
freedom for nothing.

**Permutation null.** One shuffle is one draw. The run refits the arm on 40
permuted stance tables and reports where the real arm falls in that distribution.
The real arm scores 0.1453; the null averages 0.1783 over a range of
0.0962–0.2179, and 1/40 permutations matched or beat it — p = 0.049.
Borderline, and one random pairing did beat the real one: read it as *the
pairing carries something*, not as an established effect.

**Fabrication check.** Every extracted record must quote a sentence that appears
verbatim in its statement; the check is run at extraction time and re-run as a
test over the committed table. All 141 records pass.

## Known limits

- **Fourteen meetings.** Nothing here reaches significance on the paired
  interval, and the effect is concentrated in two of them. The permutation null
  tests a different question (does the stance pairing matter?) and cannot rescue
  the sample size.
- **The extraction is not cutoff-safe by construction.** Scoring a 2015
  statement uses a model that has read the following decade. The task is
  retrospective description rather than prediction, which limits the damage, and
  the eval window is post-cutoff — but a stance score is not immune to the recall
  problem documented for the agent arms, and this has not been probed.
- **Thin variation in-window.** Across the 14 origins `hawk_dove` only ever takes
  −1 or 0 (four dovish readings, ten neutral). The features move on 2009-2024 —
  that is where the model fits them — but inside the eval window they are close
  to a single dovish-turn indicator.
- **Drivers are extracted but unused.** Seven dummies on ~130 training rows would
  buy variance. They stay in the table for auditing by theme.
- **No BOS.** The plan also asked for the Business Outlook Survey. Not done here:
  the BoC Valet API carries BOS as numeric quarterly series (`BOS_*`), which is a
  covariate question rather than a text-extraction one, and worth its own step
  once the text result is settled.

## Files

| file | what it does |
|---|---|
| `stance.py` | The extraction schema, prompt, and the verbatim-quote check. One LLM call per statement. |
| `extract_stance_table.py` | Runs it over every cached statement into `data/statement_stance.csv`. Resumable. |
| `features.py` | `StanceTable` — cutoff-aware feature rows, plus the seeded shuffle used by the control. |
| `text_logistic.py` | `BoCTextLogisticPredictor` — macro ± stance features, one class for all three arms. |
| `run_a2.py` | Scores the arms, checks the ablation, runs the permutation null, writes the leaderboard. |

Tests: `implementations/tests/boc_rate_decisions/test_a2_text_features.py` (20 cases).

## Reproduce

```bash
# once, if the press-release cache is empty (it is gitignored)
uv run python scripts/fetch_boc_press_releases.py

cd implementations
python -m boc_rate_decisions.a2_text_features.extract_stance_table   # 141 LLM calls, lite model
python -m boc_rate_decisions.a2_text_features.run_a2                 # no LLM calls
pytest tests/boc_rate_decisions/test_a2_text_features.py
```

`statement_stance.csv` is force-added past the `implementations/**/data/`
gitignore rule, because it is 141 LLM calls of output and regenerating it is the
only expensive step here. With it present, `run_a2` is reproducible offline and
spends nothing. No arm calls a model at predict time, so the eval spec's
`max_runs` budget is untouched.
