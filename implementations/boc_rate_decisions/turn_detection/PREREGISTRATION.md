# Pre-registration — turn-detection Steps 0 and 1

Written 2026-09-20, **before any 2023-2024 score was computed** and before the
static blend was computed on any window. What had already been seen: every
2025-2026 number in A1-A3, including a back-of-envelope oracle estimate for the
2025-26 t+1 window (~0.076 RPS). The 2025-26 half of Step 0 is therefore not
blind; the 2023-24 half is, and the decisions below are taken on the pooled
window so that the unseen half carries weight.

Nothing in this file may be edited after the results exist. Changes go in a
dated addendum at the bottom, with the reason.

## Fixed choices (no tuning anywhere)

- **Arms:** `market_implied` (A1 method, same estimator-selection rule),
  `boc_logistic_macro` (four macro features, C = 1.0, fit at origin),
  `categorical_frequency` (empirical frequencies of resolved meetings).
  No LLM arm: an LLM has read 2023-24, so it cannot be scored there.
- **Score:** RPS, unnormalised Epstein/Murphy convention, as the harness.
- **Windows:** 2023-24 = the 14 meetings 2023-04-12 … 2024-12-11
  (`specs/boc_rate_direction_2023_2024.yaml`); 2025-26 = the 14 meetings of the
  canonical eval spec; pooled = both, 28 meetings.
- **Missing quotes:** an origin the futures cannot price is dropped and listed.
  If more than 3 of the 14 new origins drop, Step 1 is declared
  **inconclusive**, not passed or failed.

## Step 0 — how much can a router win?

- **Oracle router:** per meeting, the lower RPS of market and logistic. Upper
  bound on any rule that picks between the two.
- **Static blend:** linear pool, `0.5 · market + 0.5 · logistic`, weight fixed
  at 0.5 and never fitted.
- **Headroom** = RPS_market − RPS_oracle.
  **Captured share** = (RPS_market − RPS_blend) / headroom.

**Decision, on the pooled 28 meetings:**

1. Captured share ≥ 70% → **drop the router direction**; a fixed blend already
   takes most of what a perfect router could.
2. Headroom < 0.02 RPS → **drop the router direction**; there is too little to
   win for an agent to justify its cost. (Added here, not in the chat plan: a
   70% rule is meaningless if the pie is tiny.)
3. Otherwise → the router direction proceeds to Step 2.

Reported, not decided on: the same numbers per window, and the blend weight
curve from 0 to 1 (descriptive only — picking a weight from it would be fitting
on the eval data).

## Step 1 — does the structural finding replicate?

The finding from A3 (2025-26, t+1): *the market is the worst of the three arms
on meetings where the Bank held, and the best on meetings where it moved.*

**Replicates on 2023-24** only if **both** hold on that window alone:

- (a) on the hold subset, `market_implied` has the highest mean RPS of the three
  arms;
- (b) on the move subset (cuts and hikes together), `market_implied` has the
  lowest mean RPS of the three arms.

Either failing → **falsified**, and the "market's whole edge is turning points"
narrative is withdrawn in the experiment log.

Reported alongside, not decided on: logistic skill vs market inside each subset,
and the same split on the pooled window.

## Addenda

### 2026-09-21 — Step 2 added (written before any P(move) number existed)

Step 2 turns the three-way direction task into the binary question the turn
detector will actually answer, and checks whether the market's move probability
is biased.

**Task.** `P(the Bank moves)` at each meeting — a cut or a hike, either
direction. Scored with the Brier score, on the same 28 meetings as Steps 0-1.

**Arms** (all cutoff-safe, no LLM):
- `market_implied` — `p_cut + p_hike` from the committed futures tables.
- `boc_logistic_macro` — `1 - p_hold` from the same fit-at-origin logistic.
- `historical_frequency` — the share of past meetings that moved, at each origin.

**Reported, not decided on:** Brier per arm per window, skill against the
market, and the Murphy decomposition (reliability / resolution / uncertainty).

**Calibration claim.** A3 suggested the market carries standing insurance
against a move, which would make its `P(move)` biased high. The claim counts as
supported only if the market's mean predicted `P(move)` exceeds the realised
move frequency **in both windows separately**. One window alone is not enough.

**Recalibration test.** If and only if the bias is supported in both windows: fit
a one-parameter correction (logistic regression of the outcome on the market's
log-odds — Platt scaling, two coefficients) on one window and score it on the
other, **in both directions**. It counts as beating the market only if the
corrected forecast has a lower Brier score than the raw market **in both
directions**. Fitting and scoring on the same window is not reported at all.

No threshold here is a kill rule: Step 2 builds an instrument, and an
instrument is not something to pass or fail. The claims above are pass/fail.


### 2026-09-21 — calibration test added (written before any recalibrated score existed)

Step 2 showed the logistic already *discriminates* better than the market at
t+2 (resolution 0.054 against 0.046) and loses only on honesty (reliability
0.036 against 0.009). Swapping the two reliability terms arithmetically would
put it at 0.132 against the market's 0.155. This tests whether that is
reachable in practice, with no LLM.

**Method.** Platt scaling (logistic regression of the outcome on the forecast's
log-odds, two parameters) fitted on one window's P(move) forecasts and scored on
the other window's. Both directions. In-sample numbers are never reported.

**Windows.** t+2 targets, now available for both: 14 from the 2023-24 origins
(targets 2023-06-07 … 2025-01-29) and 13 from the 2025-26 origins (targets
2025-03-12 … 2026-09-02). The same test is also run at t+1 for comparison.

**Claim.** The calibrated logistic beats the market at t+2 only if its Brier is
lower than the raw market's **in both directions** (fit on 2023-24 → score
2025-26, and fit on 2025-26 → score 2023-24). One direction is not enough.

**Negative control.** The same recalibration is applied to the *market's* own
forecasts. Step 2 found the market already calibrated, so a correction should
change its score very little. A large "improvement" there would mean the
procedure is fitting noise, and the logistic result would have to be discarded
too.

**If it passes,** the calibrated logistic — not the raw market — becomes the bar
any agent has to beat, and the agent's job narrows to adding resolution on top
of it.


### 2026-09-21 — market-error test added (written before any error correlation existed)

The regime check showed statements carry information about the Bank's coming
activity that its own move history lacks. It did **not** show the market lacks
it — the market reads the same statements. This tests that directly.

**Question.** Does the language variable predict the market's own signed error?

**Signed error** for meeting *i* is ``outcome_i - market_P(move)_i``. Positive
means the market under-forecast the move. If language predicts this, the market
has not priced what the statements say.

**Language variable (primary): ``lex_tilt``** — the contamination-free word-count
tilt. Chosen because it is the strongest contamination-free performer in the
regime check; that selection was made on a different outcome (forward activity
rate, not market error), but it is still a selection and is recorded here. The
other seven variables are reported as exploratory only.

**Cutoff.** The statement used at each origin is the most recent one published
on or before it — the previous meeting's, at a 28-day lead.

**Sample.** 28 meetings at t+1 and 27 at t+2, in two windows of ~14. This is
badly underpowered: it can detect only a large effect, and a null result is
therefore **not** evidence that no edge exists. Recorded here so the result is
not read as stronger than it is.

**Claim — both parts must hold:**

1. **Sign consistency.** The correlation between ``lex_tilt`` and the signed
   error has the **same sign in both windows**. (The insurance hypothesis died
   on exactly this test, so it is the one that matters at this sample size.)
2. **Out-of-sample correction.** A one-variable correction fitted on one window
   and applied to the other lowers the Brier score against the raw market **in
   both directions**.

**Negative control.** The same correction is fitted using a **shuffled**
language column (statements randomly reassigned to meetings, seeded). It must
not produce an improvement. If it does, the procedure is fitting noise.

**If both parts pass,** there is something in the statements the market has not
priced, and an agent reading them has a defined job. **If either fails,** the
honest conclusion at this sample size is "not detectable here", not "no edge".


### 2026-09-21 — lexicon v2 frozen (written before it touched the held-out window)

``lexicon_v2.py`` was developed under the protocol above: **only statements
published before 2023**, which is the half with no market data and therefore no
way to see the outcome the market test scores. Three mechanical changes, none
chosen by checking a result — context resolution for ambiguous direction words
("lower unemployment" is hawkish, "lower inflation" is dovish), negation
handling within three tokens, and larger domain-vocabulary lists with phrase
support.

**Development result** (103 meetings before 2023, forward-activity target, the
same measure the regime check used): tilt gain 0.311 → 0.326 on the overlapping
sample and 0.311 → **0.422** on the non-overlapping one. v2 is therefore the
version taken forward.

**Frozen here.** No term, window or rule may be changed after this point
without a new dated entry saying so.

**Re-run of the market-error test.** Same two conditions, same negative control,
same windows. The only change is the primary variable: ``lex2_tilt`` in place of
``lex_tilt``. This is the **second** look at the 2023-26 window and it is
pre-committed to a single variable; the v1 columns are reported alongside for
comparison but carry no claim. If the second look also comes back inconclusive,
the honest reading is that this sample cannot settle the question, and further
lexicon work on it would be fitting noise rather than measuring better.


### 2026-09-21 — turn agent frozen (written before any score was computed)

The agent in ``turn_agent.py`` was developed against label-free criteria only
(``run_agent_labelfree.py``), never against a score. Its first draft passed all
four on the post-cutoff window:

| check | result |
|---|---|
| not degenerate | p_move std 0.259, six distinct values |
| evidence grounded | 26 of 26 quotes verbatim |
| the text is used | withholding the statement moves p_move by 0.221 |
| not just recall | six-year false date + stripped dates moves it 0.093, rank corr 0.821 |

The same checks on **2023-2024** answer differently and instructively:
withholding the statement moved p_move by **0.000**, because the model recited
the withheld statement from memory. Same prompt, same model — on memorised
meetings the text is redundant, on unseen meetings it does the work. That is
the contamination mechanism, measured rather than assumed, and it is why only
the post-cutoff window is scored.

**Frozen here.** No prompt wording, schema field, or context block may change
before scoring. Temperature is 0, so the scored run is the run already made.

**Scoring.** Brier on P(move), 14 meetings from 2025-01-29 to 2026-09-02, with
the market's ``p_cut + p_hike`` as reference and the two label-free arms scored
alongside as controls.

**No significance is available.** Fourteen meetings with four moves cannot
separate these methods; every interval in A1-A3 covered zero and this will too.
The verdict below is a go/no-go for continuing, not a result.

**Worth continuing only if all three hold:**

1. the agent's Brier is below the market's (point estimate);
2. the ``no_statement`` arm scores **worse** than the agent — otherwise the
   statement is decoration and the agent is a wrapper around the rate history;
3. the evidence quotes stay verbatim at the scored temperature.

**If (1) fails but (2) holds,** the text is doing work but not enough; the
honest next step is the yield-proxy sample, not prompt iteration.

**If (2) fails,** stop. There is no reading happening and no amount of prompt
work on fourteen meetings will show otherwise.


### 2026-09-21 — live forecast series opened

Every score in this folder comes from a window the model may have read, and the
label-free checks measured what that is worth: on 2023-24 the agent recited a
withheld statement from memory. The only cure is to forecast before the outcome
exists.

**The series.** ``live_forecast.py`` appends one record per announcement,
written on or before that announcement's origin. Each record carries the
market's implied probabilities, the frozen agent's assessment with its
evidence, the no-statement control, and the lexicon reading. The script refuses
to write a record for a meeting that has already happened.

**First record: origin 2026-09-30, for the 2026-10-28 announcement.**

**Rules, fixed now:**

- **Append only.** A record is never edited. A mistaken run is superseded by a
  later record with the same origin and both stay in the file.
- **No tuning between records.** The prompt is frozen. Changing it starts a new
  series under a new name rather than continuing this one; a series whose
  method changed partway measures nothing.
- **No interim scoring against the accumulating record.** Scores may be read,
  but they may not be used to adjust anything in this folder. The point of the
  series is destroyed the moment it becomes a tuning signal.
- **No claim before 20 records.** At eight announcements a year that is about
  two and a half years. Stating the number in advance is the only defence
  against reading a lucky first five as a result.

This is deliberately slow. It is also the only evidence here that will not need
a caveat about contamination.


### 2026-09-21 — long-sample text test (written before it was run)

Everything so far has been decided, or failed to be decided, on 28 meetings.
The futures reference cannot go back further — CORRA futures start in 2023 —
but the question underneath does not actually require futures. It requires a
*market-priced* baseline, and the 2-year Government of Canada yield has been in
this repo's own data since 1991.

**Question.** Does the frozen lexicon add information about whether the Bank
moves, **beyond** a fit-at-origin model of market-priced and macro conditions,
across the full 2009-2026 sample?

This is a weaker claim than "beats the futures market" and is labelled as such
wherever it is reported. It is the only version of the question with enough
meetings to answer.

**Arms**, all binary P(move), all fit at each origin on meetings resolved
before it, all cutoff-safe (no LLM anywhere):

- ``macro`` — the four existing features, including the 2-year yield spread.
- ``macro_plus_text`` — the same plus ``lex2_tilt``, ``lex2_intensity``,
  ``lex2_hedging`` from the frozen v2 lexicon.
- ``macro_plus_shuffled`` — the same three columns, statements randomly
  reassigned to meetings (seed 42). The negative control.
- ``historical_frequency`` — the floor.

**Claim — both must hold:**

1. ``macro_plus_text`` has a lower mean Brier than ``macro``, and the paired
   bootstrap interval on the difference **excludes zero**.
2. ``macro_plus_shuffled`` does **not** show a significant improvement over
   ``macro``. If scrambled text also "helps", the gain is degrees of freedom
   rather than language.

**Reported, not decided on:** the same comparison split by era (2009-2014,
2015-2019, 2020-2022, 2023-2026), to see whether any gain is spread across
regimes or concentrated in one. A gain confined to a single era is a weaker
finding than the headline number would suggest, and this project has now been
caught by exactly that twice.

**If it passes,** the text channel beats a market-priced baseline on a sample
large enough to mean something, and the agent's unresolvable 0.0026 on fourteen
meetings becomes a plausible small version of a real effect.

**If it fails,** the agent's margin is most likely noise, and the honest move is
to a use case with more events rather than more prompt work here.


### 2026-09-21 — multi-sample averaging test (written before it was run)

The frozen agent over-forecasts: mean 0.379 against a realised 0.286, and a
reliability of 0.053 against the market's 0.009. Resolution is where it wins;
calibration is where it gives the win back.

**What is not allowed.** Shrinking the agent's probabilities toward the base
rate by a factor chosen because we observed a 9-point bias would be fitting a
correction to the fourteen meetings that revealed it — the same move that
failed three times already. That option is excluded, not deferred.

**What is allowed.** Averaging several samples reduces variance for reasons
that hold before seeing any data, and K-sample averaging is one of the three
components of the Bayesian Linguistic Forecaster. So: **K = 5 samples at
temperature 0.7, averaged.** Both numbers are fixed here, chosen by convention
rather than by trying values — no other setting will be tested.

**Prior expectation: this probably does nothing.** The repo has already found
the BoC agent deterministic at default settings and concluded that ensembling a
tool-less agent is pointless. Raising the temperature deliberately is a
different manoeuvre, but the prior is negative and is recorded here so a null
is not read as a surprise.

**Measurement.** Once, on the same fourteen t+1 meetings, against the frozen
single-sample agent and the market.

**Adoption rule, fixed now.** The averaged version replaces the single-sample
agent in the live series only if **both**:

1. its Brier is lower by more than **0.005** — a bare improvement on fourteen
   meetings is noise, so the threshold is deliberately above that; and
2. the no-statement contrast survives: removing the statement must still cost
   the averaged agent more than 0.05 Brier.

This is a one-bit decision taken on the evaluation window, which is a weaker
form of the thing this file exists to prevent. It is accepted because the
alternative — adopting a procedure blind and carrying it for two and a half
years of live records — is worse, and because one bit from a pre-declared
threshold is not a tuning loop. It is recorded as a known compromise.
