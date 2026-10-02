# Turn detection — Steps 0 and 1

> The plan that came out of A1-A3: the market's whole advantage is turning
> points, so the useful agentic role is deciding *when* a meeting is live, not
> producing a probability for every meeting.
> Judgement criteria were fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md)
> before any number below existed.

## Step 1 — the structural finding replicates

A3 found, on 2025-26, that the market is the **worst** of the three cutoff-safe
arms on meetings where the Bank held and the **best** on meetings where it
moved. That rested on 14 meetings with 4 moves. Step 1 re-ran the split on
**2023-2024** — 14 meetings never scored before, a different regime, and the
only two hikes available anywhere in this project.

**2023-2024 (7 moves, 7 holds), mean RPS**

| arm | on the 7 moves | on the 7 holds |
|---|---|---|
| market_implied | **0.189** ← best | 0.109 ← **worst** |
| boc_logistic_macro | 0.545 | **0.007** ← best |
| categorical_frequency | 0.852 | 0.025 |

Both pre-registered conditions pass. **VERDICT: REPLICATED.**

Pooled over all 28 meetings (11 moves, 17 holds) the pattern is the same:
market 0.195 / 0.086, logistic 0.530 / 0.018, climatology 0.846 / 0.027.

This is now the most solid result in the BoC work: two windows, two regimes,
cuts and hikes, no LLM anywhere, and a rule written down before the second
window was scored.

## Step 0 — there is room, and averaging cannot reach it

| window | market | logistic | blend 0.5 | oracle | headroom | captured share |
|---|---|---|---|---|---|---|
| 2023-2024 | 0.149 | 0.276 | 0.182 | 0.098 | 0.052 | −63% |
| 2025-2026 | 0.108 | 0.162 | 0.123 | 0.070 | 0.038 | −41% |
| **pooled** | **0.129** | 0.219 | 0.153 | **0.084** | **0.045** | **−53%** |

**VERDICT: PROCEED to Step 2.** Headroom is 0.045 RPS — a perfect router would
score 35% better than the market — comfortably above the 0.02 kill line.

**The fixed blend fails, and that is the informative part.** Averaging the two
forecasts is *worse than the market alone* (0.153 vs 0.129), and the weight
curve is monotone: every weight short of 100% market is worse than the market.
The reason is exactly the structure Step 1 confirms — the two are good in
disjoint places, so a pool dilutes the market's sharp move calls with the
logistic's confident holds and vice versa. It is worse than both in *both*
subsets. **You have to choose, not average**, which is precisely the job a
router does and a blend cannot.


## Step 2 — the instrument, and one hypothesis killed

The turn question on its own: **P(the Bank moves)**, cut or hike, scored with
Brier over the same 28 meetings. Every arm is a projection of something that
already exists — the market's `p_cut + p_hike`, the logistic's `1 - p_hold`,
and the historical move rate — so nothing new is being forecast, only measured
differently.

| window | market | logistic | climatology |
|---|---|---|---|
| 2023-2024 | **0.149** | 0.263 (−0.76) | 0.337 (−1.25) |
| 2025-2026 | **0.108** | 0.159 (−0.47) | 0.209 (−0.93) |
| **pooled** | **0.129** | 0.211 (−0.64) | 0.273 (−1.12) |

On the binary question the market's dominance is **larger**, not smaller. That
is the point of the instrument: on the three-way task, ten of fourteen meetings
are holds everyone gets right, which flattered the competition.

### The market is not selling insurance

A3 suggested the market carries standing insurance against a move, which would
make its `P(move)` biased high and a simple recalibration enough to beat it.
Pre-registered test: the bias must be positive **in both windows**.

| window | mean forecast | realised move rate | bias |
|---|---|---|---|
| 2023-2024 | 0.479 | 0.500 | **−0.021** |
| 2025-2026 | 0.304 | 0.286 | **+0.018** |

Opposite signs. **VERDICT: NOT SUPPORTED** — and the recalibration test was
skipped by the pre-registered rule rather than run and quietly ignored. Pooled
bias is −0.001 and pooled reliability is 0.0012: the market's move odds are
about as well calibrated as 28 observations can show.

**The cheap path is closed.** There is no market bias to correct, so beating it
requires information it does not have — which is what an agent was always for,
but there is now no shortcut to it.

### What that says about the logistic

| pooled | brier | reliability ↓ | resolution ↑ | mean forecast | bias |
|---|---|---|---|---|---|
| market | 0.129 | **0.0012** | **0.1105** | 0.392 | −0.001 |
| logistic | 0.211 | 0.0485 | 0.0827 | 0.213 | **−0.180** |
| climatology | 0.273 | 0.0301 | 0.0000 | 0.219 | −0.174 |

The market beats the logistic on *both* components. And the logistic's
celebrated advantage on quiet meetings is not skill — it is **bias**: it
forecasts moves at 21% against a realised 39%, so it wins every hold by being
overconfident and loses every move for the same reason.

This refines Step 0's reading. The router's 0.045 headroom is real, but it comes
from exploiting a systematically under-forecasting model on meetings where
under-forecasting happens to be right — not from blending two complementary
skills. What is actually wanted is **resolution**: a forecast that separates
live meetings from quiet ones, and is calibrated while doing it.

### Where resolution is scarce at t+2 — WITHDRAWN

The first version of this section reported, from the 2025-26 window alone, that
the market's resolution more than halves at t+2 (0.109 → 0.046) while the
logistic's nominally exceeds it, and concluded that t+2 was the opening worth
attacking. **Building the second window withdrew that.**

| horizon | window | market resolution ↑ | logistic resolution ↑ |
|---|---|---|---|
| t+1 | 2023-2024 | 0.098 | 0.068 |
| t+1 | 2025-2026 | 0.109 | 0.087 |
| t+2 | 2023-2024 | **0.150** | 0.050 |
| t+2 | 2025-2026 | 0.046 | 0.054 |

In 2023-24 the market's t+2 resolution (0.150) is *higher* than its own t+1
resolution, and three times the logistic's. The "collapse at t+2" was a
one-window artifact, and so was "the logistic discriminates better there".
Neither claim survives its second window.

What is left after both windows: the market beats the logistic on resolution in
three of four horizon/window cells, and on reliability in all four.

## The calibration test — failed, and it explains why

Step 2's arithmetic suggested a free win: give the logistic the market's
honesty while keeping its own discrimination and it would score 0.132 against
the market's 0.155 at t+2. [`run_calibration_test.py`](run_calibration_test.py)
tested that with a two-parameter Platt correction fitted on one window and
scored on the other, both directions.

| horizon | fitted on | scored on | logistic raw | logistic calibrated | market |
|---|---|---|---|---|---|
| t+2 | 2023-24 | 2025-26 | 0.179 | **0.368** | 0.155 |
| t+2 | 2025-26 | 2023-24 | 0.318 | **0.361** | 0.168 |
| t+1 | 2023-24 | 2025-26 | 0.159 | **0.301** | 0.108 |
| t+1 | 2025-26 | 2023-24 | 0.263 | 0.232 | 0.149 |

**VERDICT: fails at both horizons**, and not narrowly — the correction usually
makes things *worse*, roughly doubling the score.

The **negative control** says why it should not be trusted even where it helps:
recalibrating the already-calibrated market moves its Brier by up to 0.049,
which is not the near-zero a sound procedure would give. By the
pre-registration, that alone voids the result.

### The reason, and it is the interesting part

The two windows have completely different base rates:

| window | share of meetings that moved |
|---|---|
| 2023-2024 | **57%** |
| 2025-2026 | **23%** |

A correction fitted where the Bank moved at 57% of meetings cannot be carried
into a period where it moved at 23%. **Calibration does not transport across
regimes.**

And that reframes the market's advantage. It is not only that the market
discriminates well — it is that it **re-estimates how active the Bank currently
is, continuously, from prices**. Its reliability stays near zero across windows
whose base rates differ by a factor of 2.5 (0.0036 and 0.0085 at t+1). No
statically fitted model can do that, and no correction learned from history can
fake it.

So the gap is not "the logistic is dishonest and needs fixing". It is that a
fit-at-origin model has no way to know the regime changed, and the market does.

## What a router would actually have to achieve

Descriptive design guidance, not a pre-registered rule. Pooled window; the bar
is the market's 0.1287.

```
             sens=0.5  sens=0.6  sens=0.7  sens=0.8  sens=0.9  sens=1.0
spec=1.0       0.1533    0.1401    0.1269    0.1138    0.1006    0.0874
spec=0.9       0.1574    0.1442    0.1311    0.1179    0.1047    0.0916
spec=0.8       0.1615    0.1483    0.1352    0.1220    0.1089    0.0957
spec=0.7       0.1656    0.1525    0.1393    0.1261    0.1130    0.0998
```

Two numbers set the design:

- **Catch about 70-80% of moves** and the router beats the market, almost
  regardless of how many false alarms it raises.
- **Missing a move costs 4.9× what a false alarm costs** (+0.335 RPS against
  +0.068). The agent should be biased toward calling a meeting live; caution is
  the expensive error here, not eagerness.

## Why no LLM appears in either step

Every arm is cutoff-safe: futures settlements, a four-feature logistic, and
counting. That is deliberate — the models in use have read 2023-24, so no text
or agent arm can be scored there, and the replication would be worthless if it
depended on one. The cost is that Step 1 says nothing about whether text helps
at more turns; that question stays inside the 2025-26 window until a
contamination-free text feature exists.

## Files

| file | what it does |
|---|---|
| `PREREGISTRATION.md` | The judgement criteria, fixed before the results (Step 2 added as a dated addendum). Tests assert the code still matches it. |
| `ceiling.py` | Pool, oracle, headroom, and the router break-even grid. |
| `run_steps.py` | Steps 0-1: scores both windows, applies both verdicts, writes the CSVs. |
| `move_task.py` | Step 2: the P(move) target, the arms' projections, Brier, the Murphy decomposition and Platt scaling. |
| `run_step2.py` | Step 2: leaderboard, calibration, the bias verdict and the t+2 panel. |
| `run_calibration_test.py` | The out-of-sample calibration test at both horizons, with the market as negative control. |
| `regime_language.py` | Does language predict coming activity beyond the trailing move rate? |
| `lexicon.py` | The contamination-free word-count counterpart of A2's stance scores (v1). |
| `lexicon_v2.py` | Context-aware, negation-aware, phrase-aware refinement; developed on pre-2023 only, then frozen. |
| `run_lexicon_dev.py` | Compares v1 and v2 on the development split, and refuses to touch the held-out window. |
| `run_regime_check.py` | Runs the regime check over both measures and both samples. |
| `run_market_error_test.py` | Does language predict the market's signed error? Both horizons, with a shuffled-language control. |
| `turn_agent.py` | The agent: regime call, P(move), verbatim evidence. Frozen before scoring. |
| `run_agent_labelfree.py` | Stage 1 — prompt development against criteria that read no outcome. |
| `run_agent_score.py` | Stage 2 — scores the frozen agent and its controls against the market. |
| `live_forecast.py` | Appends one prospective record per announcement, written before the decision. |
| `score_live.py` | Scores whatever the live series has resolved; states nothing before 20 records. |
| `long_sample.py` / `run_long_sample.py` | Does the lexicon add over a market-priced baseline across 141 meetings? |
| `specs/boc_rate_direction_t2_2023_2024.yaml` | The 2023-24 window at the t+2 horizon, built so t+2 has two windows. |
| `specs/boc_rate_direction_2023_2024.yaml` | The 14-origin replication window. |

`implementations/**/data/` is gitignored, so the result files here are
force-added: they are findings and LLM outputs, not fetched data. The one that
matters most is `live_forecasts.jsonl` — a prospective record that is not in
version control is not evidence.

Outputs in `data/`: `step0_ceiling.csv`, `step1_replication.csv`,
`weight_curve.csv`, `router_requirement.csv`, `per_meeting_rps_pooled.csv`,
`step2_move_leaderboard.csv`, `step2_calibration.csv`, `step2_t2_panel.csv`,
`calibration_test.csv`, `calibration_baseline.csv`, `regime_language_results.csv`,
`regime_language_frame.csv`, `market_error_correlations.csv`,
`market_error_corrections.csv`, `lexicon_dev_comparison.csv`, `agent_labelfree_*.csv`, `agent_score.csv`, and the market tables for both windows at both
horizons.

Tests: `implementations/tests/boc_rate_decisions/test_turn_detection.py` (24 cases).

## Reproduce

```bash
cd implementations
# once, if the futures cache does not reach back to 2023:
python -m boc_rate_decisions.a1_market_implied.fetch_market_data \
    --start 2023-01-01 --end 2026-09-20 --force
python -m boc_rate_decisions.turn_detection.run_steps
python -m boc_rate_decisions.turn_detection.run_step2
python -m boc_rate_decisions.turn_detection.run_calibration_test
python -m boc_rate_decisions.turn_detection.run_regime_check
python -m boc_rate_decisions.turn_detection.run_lexicon_dev
python -m boc_rate_decisions.turn_detection.run_market_error_test
pytest tests/boc_rate_decisions/test_turn_detection.py
```

No LLM calls, no eval-spec budget.


## The regime check — text does carry something the move history does not

The calibration test said the gap is a **base rate** gap: the market keeps
re-estimating how active the Bank currently is, and nothing fitted to history
can. The cheap question before building anything: is that visible in the
statements?

Setup: for each of 129 meetings from 2009 to 2026, predict the share of the
**next four** meetings that produce a move. Baseline predictor: the share of the
**previous eight** that did. Then add one language variable and see whether R²
moves. Both an overlapping sample and a non-overlapping one (every fourth
meeting) are reported, since neighbouring forward windows share meetings.

| variable | source | R² alone | R² + variable | gain |
|---|---|---|---|---|
| *(trailing move rate only)* | — | **0.033** | — | — |
| stance intensity | LLM (A2) | 0.033 | 0.224 | **+0.191** |
| lexicon tilt | word list | 0.033 | 0.169 | **+0.136** |
| lexicon intensity | word list | 0.033 | 0.055 | +0.022 |
| hedging words | word list | 0.033 | 0.052 | +0.019 |
| guidance level | LLM (A2) | 0.033 | 0.034 | +0.001 |

*(non-overlapping sample, n = 33; the overlapping sample agrees in direction.)*

Two things stand out.

**The trailing move rate is nearly worthless.** R² = 0.033. Knowing the Bank
moved at six of the last eight meetings tells you almost nothing about the next
four. That is precisely why a fit-at-origin model cannot track the regime — the
history it fits on does not contain the answer.

**The text does better, and it is not an LLM artifact.** A2's stance scores come
from a model that has read 2009-2026, so a relationship found with them proves
nothing on its own. [`lexicon.py`](lexicon.py) measures the same thing
mechanically — fixed word lists, counted, no model — and the relationship
survives at roughly two-thirds the strength. Hedging words, included as a
sanity check, show almost nothing, so the measure is not simply rewarding long
statements.

### What this does and does not establish

- ✅ Statements carry information about the Bank's **coming activity level**
  that its own recent behaviour does not.
- ✅ It survives a contamination-free measure, so it is not recall.
- ❌ **It does not show the market misses this.** The market reads the same
  statements. This compares text against move history, not against the market.
- ⚠️ n = 33 on the conservative sample, and no significance test was run. Treat
  it as a reason to continue, not as a result.

The next question is therefore sharper and still free: **does the language
variable predict the market's own errors?** If it does not, the market has
already priced it and there is nothing here for an agent to find.


## The market-error test — direction consistent, correction does not transport

The regime check compared text against the Bank's move history. This compares
it against the market: for each meeting, does the language of the last
statement predict the market's **signed error**, `outcome − market_P(move)`?
Positive error means the market under-forecast the move.

Pre-registered: the primary variable is `lex_tilt` (contamination-free), both
conditions must hold, and a shuffled-language arm runs as negative control.

**Condition 1 — same sign in both windows: PASSES at both horizons.**

| horizon | 2023-2024 | 2025-2026 |
|---|---|---|
| t+1 | −0.109 | −0.405 |
| t+2 | **−0.225** | **−0.233** |

All four cells negative, and the two t+2 values land within 0.01 of each other
across regimes that could hardly be more different (57% move rate against 23%).
Of the three lexicon variables, only `lex_tilt` — the pre-registered one — is
sign-consistent; `lex_intensity` and `lex_hedging` both flip. Choosing the
primary in advance is what makes that observation worth anything.

**Condition 2 — out-of-sample correction beats the market both ways: FAILS.**

| horizon | fitted on | scored on | market | corrected |
|---|---|---|---|---|
| t+1 | 2023-24 | 2025-26 | 0.1080 | **0.1039** ✓ |
| t+1 | 2025-26 | 2023-24 | 0.1494 | 0.1571 ✗ |
| t+2 | 2023-24 | 2025-26 | 0.1552 | 0.1612 ✗ |
| t+2 | 2025-26 | 2023-24 | 0.1675 | **0.1461** ✓ |

One direction each. Not both.

**Negative control: clean.** Shuffled language improved 0 of 2 directions at
both horizons, so the procedure is not simply fitting whatever it is handed.

### Verdict and what it means

**NOT DETECTABLE AT THIS SAMPLE SIZE** — the pre-registered wording, and the
honest one. With fourteen meetings per window the sign of an effect can be
stable while its magnitude is not, which is exactly the pattern here: the
direction replicates, the slope does not transport.

What this is **not**: evidence that the market has priced everything. A null on
28 meetings rules out only a large effect.

What it is: a consistent direction, a clean control, and a measurement crude
enough that improving it is the obvious next move.

### If the lexicon is refined, the protocol matters

Refining word lists after seeing this result, on this data, is how noise gets
fitted. There is a clean way to do it: **develop the lexicon only on statements
before 2023**, where no market data exists and so the outcome being tested
cannot be peeked at, validate there against the forward move rate, freeze it,
and only then re-run the market test on 2023-26. The 141-statement cache makes
that possible at no cost.


## Lexicon v2 — better measurement, same transport failure

Following the protocol above, [`lexicon_v2.py`](lexicon_v2.py) was built using
**only statements before 2023**: context resolution for ambiguous direction
words ("lower unemployment" hawkish, "lower inflation" dovish), negation
handling, and larger phrase-aware vocabulary. On the development split it beat
v1 — tilt gain 0.311 → **0.422** on the non-overlapping sample — and was frozen
before touching the held-out window.

**It measures better.** Correlation with the market's signed error:

| horizon | | 2023-2024 | 2025-2026 |
|---|---|---|---|
| t+1 | v1 | −0.109 | −0.405 |
| t+1 | **v2** | **−0.106** | **−0.130** |
| t+2 | v1 | −0.225 | −0.233 |
| t+2 | **v2** | **−0.384** | **−0.378** |

At t+2 the correlation nearly doubles and the two windows land within 0.006 of
each other. That is what a sharper instrument looks like.

**The correction still does not transport.** Both directions worse at t+2, both
worse at t+1. The negative control stays clean (0 of 2).

### Why, and why more lexicon work will not fix it

The correction is `p_market + (a + b·language)`. The intercept `a` absorbs the
mean error of the window it was fitted on — and the two windows have move rates
of 57% and 23%. Fitting where the Bank moved constantly and applying where it
mostly sat still ships that base rate along with the correction.

This is the **third** time transport has failed here, always for the same
reason: [the Platt recalibration](#the-calibration-test--failed-and-it-explains-why),
the v1 correction, and now the v2 correction. The binding constraint is not
measurement quality — v2 demonstrably measures better and transported no better.
It is that two windows of fourteen meetings, in opposite regimes, cannot support
a fitted mapping between them.

The pre-registration committed in advance to the reading: **this sample cannot
settle the question, and further lexicon work on it would be fitting noise
rather than measuring better.**

### What would actually unlock it

More meetings with a market reference. The futures only start in 2023, but the
**2-year Government of Canada yield** in the repo's own data reaches back to
1991. A crude market-expectation proxy built from it would extend the sample
from 28 meetings to roughly 130 and, more importantly, across five or six
regimes instead of two — enough for transport to mean something. It is a worse
reference than futures; it is available for a hundred more meetings.


## The turn agent — first thing in this project to beat the market, and it still is not significant

The agent ([`turn_agent.py`](turn_agent.py)) answers two questions in order:
what kind of period is the Bank in, then what is the probability it moves at
this meeting. It quotes the statement verbatim so the assessment can be checked
against its source. It is **not** told that misses cost five times what false
alarms do — that asymmetry belongs to a decision downstream of the probability,
and Brier rewards honest belief.

### Stage 1: the prompt was developed without ever seeing a score

Fourteen scoreable meetings cannot support prompt tuning, so development used
label-free criteria only ([`run_agent_labelfree.py`](run_agent_labelfree.py)).
The first draft passed all four and was frozen.

The same checks on the two windows say something worth keeping:

| check | 2023-2024 (memorised, n=3) | 2025-2026 (post-cutoff, n=14) |
|---|---|---|
| withholding the statement moves p_move by | **0.000** | **0.221** |
| evidence verbatim | 6/6 | 26/26 |
| p_move spread (std) | 0.21 | 0.26 |

*(the 2023-24 column is a three-meeting probe, not a full run — it was stopped
once the zero appeared, because a zero on three meetings already shows that the
check cannot work on a window the model has memorised)*

On 2023-24 the statement is worth exactly nothing — because the model recites
its content from memory when it is withheld, naming the March 2023 conditional
pause and the June 2023 hike unprompted. Same prompt, same model: **where
memory is available the text is redundant, where it is not the text does the
work.** That is the contamination mechanism measured directly, and it is why
only the post-cutoff window is scored.

### Stage 2: scores on the 14 post-cutoff meetings

| arm | Brier ↓ | reliability ↓ | resolution ↑ | mean forecast | vs market (95% CI) |
|---|---|---|---|---|---|
| **agent** | **0.1054** | 0.0526 | **0.1469** | 0.379 | −0.0026 (−0.064, +0.060) |
| market | 0.1080 | **0.0085** | 0.1088 | 0.304 | — |
| agent, false date | 0.1725 | 0.0357 | 0.0612 | 0.300 | +0.065 (−0.020, +0.153) |
| agent, no statement | 0.2611 | 0.0948 | 0.0327 | 0.571 | +0.153 (+0.011, +0.304) |

*(realised move rate 0.286)*

**The agent beats the market, and the margin means nothing.** −0.0026 Brier
against a confidence interval twenty times wider. Fourteen meetings with four
moves cannot do better than that, which was stated before the run.

**What the mechanism says is more interesting than the margin.** The agent's
resolution is 0.147 against the market's 0.109 — it separates live meetings
from quiet ones *better* than the futures curve does. It gives that back on
calibration (0.053 against 0.009; it over-forecasts moves, 38% against a
realised 29%). The whole investigation predicted the job was to add resolution;
the agent added resolution.

### The controls are significant, and that is the real result

| comparison | Δ Brier | 95% CI | |
|---|---|---|---|
| agent with statement vs without | **−0.156** | (−0.287, −0.026) | **significant** |
| agent with real dates vs false dates | −0.067 | (−0.143, −0.004) | significant |

"Agent beats market" is not establishable here. **"The statements are doing
real work inside the agent" is** — the first significant text result anywhere in
this project. Without the statement the agent degenerates to forecasting 57% at
every meeting and its resolution collapses to 0.033.

The date control is significant too but is **confounded**: shifting six years
forward lands in 2031-2032, so the degradation could be lost recall or simply a
model handling an implausible date badly. It is not clean evidence either way.

### What not to do next

The agent over-forecasts by 9 percentage points. Correcting that on these
fourteen meetings would be the fourth time this folder fits a mapping to a
window too small to support one, and the first three all failed. The fix
belongs after the sample grows, not before.



## The agent at t+2 — significantly worse, and it says why

The same frozen agent, asked about the announcement after next (about ten weeks
out) instead of the next one. All four label-free checks still pass: 24 of 24
quotes verbatim, five distinct probabilities, withholding the statement still
moves the answer by 0.100.

The score does not.

| arm | Brier ↓ | reliability ↓ | resolution ↑ | mean forecast | vs market (95% CI) |
|---|---|---|---|---|---|
| **market** | **0.1552** | **0.0087** | **0.0456** | 0.277 | — |
| agent, false date | 0.2014 | 0.0346 | 0.0066 | 0.394 | +0.046 (−0.018, +0.113) |
| agent | 0.3160 | 0.1701 | 0.0237 | **0.522** | **+0.161 (+0.052, +0.267)** |
| agent, no statement | 0.3368 | 0.1776 | 0.0006 | 0.591 | +0.182 (+0.026, +0.320) |

*(13 meetings, realised move rate 0.231)*

**The agent is significantly worse than the market at t+2** — the first
significant score difference anywhere in this folder, and it goes against us.

Both halves of the t+1 result reverse:

| | t+1 | t+2 |
|---|---|---|
| agent resolution vs market | **0.147 vs 0.109** ✓ | 0.024 vs 0.046 ✗ |
| agent mean forecast vs realised | 0.379 vs 0.286 | **0.522 vs 0.231** |

At ten weeks the agent loses its discrimination *and* inflates. Asked about a
meeting far enough out that it genuinely cannot tell, it answers 52% — when the
base rate is 23%. **It converts "I am less certain" into "a move is more
likely"**, which is the wrong direction: uncertainty about a rare event should
pull toward the base rate, not away from it.

**Decision: t+2 is not added to the live series.** The original reason to add it
was to double the evidence stream at no extra waiting. Doubling a stream of a
known, significant failure is not worth the calls.


## Multi-sample averaging — no effect, for the reason predicted

The agent's weakness is calibration: it forecasts moves at 38% against a
realised 29%. Correcting that with a factor chosen *because we measured that
bias* would be fitting to the fourteen meetings that revealed it, so that option
was excluded rather than deferred. What was allowed is a procedure justified
before seeing data: **K = 5 samples at temperature 0.7, averaged**, both numbers
fixed in the pre-registration and never varied.

| arm | Brier ↓ | reliability ↓ | resolution ↑ | mean forecast | distinct values |
|---|---|---|---|---|---|
| agent, K=1 | **0.1054** | 0.0526 | 0.1469 | 0.379 | 6 |
| agent, K=5 | 0.1107 | 0.0520 | 0.1505 | 0.395 | 11 |
| no statement, K=1 | 0.2611 | 0.0948 | 0.0327 | 0.571 | — |
| no statement, K=5 | 0.2615 | 0.0831 | 0.0315 | 0.540 | — |

K=5 − K=1 is **+0.0053** (95% CI −0.033 to +0.036). Condition 1 required an
improvement above 0.005; this is a deterioration inside the noise.

**Decision: the live series keeps K = 1.**

The mechanism is exactly what the pre-registration expected. Averaging cuts
*variance*: the granularity improves (6 distinct probabilities become 11) and
resolution ticks up. It does nothing to *bias*: reliability is 0.0520 against
0.0526, and the mean forecast moves the wrong way. The agent is not noisy, it
is systematically high, and averaging a systematically high forecaster five
times gives a systematically high forecaster.

This also replicates an earlier repo finding on the BoC agent — that
ensembling a tool-less agent is pointless — by a different route.

## The long-sample test — the lexicon route is dead

Every test above ran on 28 meetings. This one runs on **141**, by swapping the
futures reference for a market-priced macro baseline whose strongest feature is
the 2-year Government of Canada yield — the bond market's own read on policy.
Weaker claim, four times the sample, no LLM anywhere.

| arm | Brier ↓ | reliability ↓ | resolution ↑ | vs macro (95% CI) |
|---|---|---|---|---|
| **macro** | **0.1716** | 0.0073 | 0.0111 | — |
| macro + shuffled text | 0.1782 | 0.0139 | 0.0154 | +0.0066 (−0.009, +0.023) |
| macro + text | 0.1799 | 0.0167 | 0.0146 | +0.0083 (−0.003, +0.019) |
| historical frequency | 0.1824 | 0.0143 | 0.0056 | +0.0108 |

*(141 meetings, 22.7% moved)*

**The text arm is indistinguishable from three columns of noise.** Adding the
real lexicon makes the model worse by 0.0083; adding the *scrambled* lexicon
makes it worse by 0.0066. Whatever the word counts carry does not survive
contact with a per-meeting forecast.

By era, it is worse in three of four:

| era | n | move rate | macro | + text | Δ |
|---|---|---|---|---|---|
| 2009-2014 | 47 | 0.106 | 0.1075 | 0.1097 | +0.0022 |
| 2015-2019 | 40 | 0.175 | 0.1420 | 0.1644 | +0.0224 |
| 2020-2022 | 24 | 0.333 | 0.1674 | 0.1785 | +0.0111 |
| 2023-2026 | 30 | 0.400 | 0.3150 | 0.3117 | −0.0033 |

**VERDICT: NOT ESTABLISHED.**

### This does not contradict the regime check, and the difference matters

The regime check found the lexicon predicts the **move rate over the next four
meetings** with a large R² gain. This test asks whether it predicts **whether
the Bank moves at one particular meeting**. Those are different questions, and
the answer is apparently yes to the first and no to the second.

Knowing "the Bank is about to enter an active stretch" does not tell you which
of the next four announcements carries the move. The regime signal is real and
too diffuse to convert, at least through a linear term in a per-meeting model.

### Where that leaves the agent

The pre-registration said a failure here would make the agent's 0.0026 margin
"most likely noise". That inference assumed the lexicon is a fair stand-in for
what the agent does with a statement. The agent's own control argues it is not:
removing the statement costs the agent 0.156 Brier, significantly, while
scrambling the lexicon costs a linear model nothing. **Word counts and a model
reading for meaning are not the same instrument**, and only one of them works
here.

That is an interpretation this pre-registration did not anticipate, so it is
labelled post-hoc. What it leaves is an impasse specific to this use case:

- the only text measure that can use the long sample does not work per meeting;
- the only text measure that works cannot use the long sample, because the
  model has read it.

No amount of further work inside this folder resolves that. Either the live
series accumulates (eight announcements a year), or the question moves to a
setting with more post-cutoff events.

## The live series, as settled

| decision | outcome |
|---|---|
| horizon | **t+1 only** — t+2 is significantly worse, so doubling that stream is not worth the calls |
| sampling | **K = 1** — averaging failed its pre-registered threshold |
| prompt | frozen, unchanged since before any score |
| first record | **origin 2026-09-30**, for the 2026-10-28 announcement |
| first claim | not before 20 resolved records |

## Next

The target has changed. Step 2 said "add resolution at t+2"; the calibration
test showed that rested on one window and does not replicate. What survived two
windows is different and more specific:

> **The market tracks how active the Bank currently is. Nothing we fit can.**

That is the gap an agent could plausibly fill, and it is a text-shaped job
rather than a numbers-shaped one: reading whether the Bank has entered a period
where it moves often. It is also honest about scope — it is a claim about the
*base rate*, not about picking which individual meeting turns.

That check has now been run — see the regime section above. Statements do carry
regime information the move history lacks, and it survives a contamination-free
word-count measure. What remains untested is whether the **market** already has
it, which is the difference between an edge and a coincidence.

That test has now been run — see the market-error section above. The direction
is consistent across all four window/horizon cells and the negative control is
clean, but the correction does not transport, so the verdict is "not detectable
at this sample size" rather than an edge.

That sharpening has now been done — lexicon v2, developed on pre-2023 only and
frozen. It measures better (the t+2 correlation nearly doubles and stabilises
across windows) and transports no better, which identifies the real constraint:
two windows of fourteen meetings in opposite regimes.

So the next move is neither an agent nor another lexicon. It is **a market
reference with a longer history** — a proxy built from the 2-year yield, which
reaches back to 1991 and would extend the sample from 28 meetings to about 130
across five or six regimes. Only then can "does this transport" be asked
meaningfully.

The 2026-09-30 origin (for the 2026-10-28 announcement) is the first chance to
record a genuinely prospective, uncontaminated forecast. The instrument now
exists, so even a market-only entry logged that day is worth having.
