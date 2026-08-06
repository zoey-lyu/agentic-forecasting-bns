"""WTI crude oil analyst agent configurations and prompt builder.

Provides four :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`
factories that define progressive agent capability levels:

1. :func:`build_wti_basic_config` — LLM reasons from price history alone (no tools).
2. :func:`build_wti_news_config` — Adds bounded Google Search via a
   :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
   sub-agent with strict temporal cutoffs.
3. :func:`build_wti_code_exec_config` — Adds Gemini native code execution and
   three forecasting skills on top of the news-grounded configuration.
4. :func:`build_wti_tool_config` — Adds a conventional
   :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
   (AutoARIMA) on top of news grounding — a rigid, pre-specified alternative to
   open-ended code execution.

Also provides:

- :class:`WtiPriceForecastPromptBuilder`: Pydantic ``BaseModel`` that serialises
  the task and history into a structured JSON payload for the agent.
- :func:`build_wti_agent_predictor`: convenience factory that wires a config to
  an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use without importing the full
predictor stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from aieng.forecasting.data import DataService
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentForecastOutput,
    AgentPredictor,
    ContinuousAgentForecastOutput,
    ForecastTool,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorSource
from energy_oil_forecasting.analyst_agent.news_cache import NewsCacheSource
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------

_WTI_MULTITASK_ANALYST_INSTRUCTION = """\
## Role

You are an expert WTI crude oil market analyst.

## Input

You will receive a JSON payload containing:
- `task_spec`: the exact question and required JSON output schema
- `as_of`: the forecast origin date (temporal cutoff)
- `origin_price_usd_bbl`: WTI close on the origin date
- `target_history_csv`: compressed WTI daily close history

When context retrieval is enabled, call ``search_web`` BEFORE answering.

## Output contract

Read the data (and briefing, if retrieved) carefully, then execute the task \
in `task_spec` precisely.

If a `set_model_response` tool is available, call it with your complete JSON \
as `json_response` — the exact schema is described in `task_spec`. Otherwise \
return the JSON directly as plain text with no preamble.\
"""


def _build_wti_analyst_instruction(*, include_search_guidance: bool = True) -> str:
    """Build the WTI analyst instruction, embedding the output schema from the class.

    Using a function instead of a static string ensures the ``## Output schema``
    block is always in sync with ``ContinuousAgentForecastOutput`` —
    no manual JSON to maintain.

    Parameters
    ----------
    include_search_guidance : bool
        When ``False``, the ``## Analysis discipline`` section describing the
        live ``search_web`` tool is omitted (callers append
        :data:`_CACHED_NEWS_SUPPLEMENT` explaining the pre-cached briefing
        instead). Mirrors :func:`_build_wti_anchored_instruction`'s parameter
        of the same name.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    instruction = (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n"
        "6. When tools are enabled, conclude with `set_model_response` to return the structured forecast.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. Omit any field not shown above.\n\n"
        "Document your key assumptions (OPEC+ policy, shipping lane risk, inventory "
        "levels, macro demand) in the `rationale` fields of your forecast output."
    )
    if not include_search_guidance:
        return instruction
    return (
        instruction + "\n\n"
        "## Analysis discipline\n\n"
        "When context retrieval is available, call ``search_web`` to gather market "
        "intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended queries (call ``search_web`` once per topic):\n"
        '- ``search_web(query="WTI crude oil price trend and OPEC+ supply decisions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="Persian Gulf geopolitical risk shipping lane disruptions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="US Strategic Petroleum Reserve policy and global demand outlook", cutoff_date=<as_of>)``'
    )


_WTI_ANALYST_INSTRUCTION = _build_wti_analyst_instruction()
_WTI_ANALYST_INSTRUCTION_CACHED_NEWS = _build_wti_analyst_instruction(include_search_guidance=False)


def _build_wti_anchored_instruction(*, include_search_guidance: bool = True, two_sided: bool = False) -> str:
    """Build the anchored-variant instruction, embedding ``AnchoredForecastOutput``.

    Parallel to :func:`_build_wti_analyst_instruction`, but rules 2-4 (which
    describe a raw point forecast and quantile grid) are replaced with the
    bounded-signal contract — the agent never emits prices directly in this
    variant, only a location signal and a width signal relative to the
    statistical anchor injected into the payload (see
    :class:`AnchoredWtiPromptBuilder`).

    Parameters
    ----------
    include_search_guidance : bool
        When ``True`` (default, used by :func:`build_wti_anchored_config`
        with live news), append the "Analysis discipline" section describing
        how to call ``search_web``. When ``False`` (used when a
        :class:`~energy_oil_forecasting.analyst_agent.news_cache.NewsCacheSource`
        supplies a pre-cached briefing instead —
        :data:`_CACHED_NEWS_SUPPLEMENT` explains that briefing instead),
        omit it: telling the agent to call a tool that isn't attached would
        be actively misleading.
    two_sided : bool, default=False
        Neutralise the directionally-loaded wording in the Role line and in
        rules 3-4, for use with :data:`_ANCHOR_SUPPLEMENT_TWO_SIDED`. In oil,
        "geopolitical risk" is a one-directional term — a risk premium only
        ever adds to price — and the default rules describe a nonzero signal
        as "deviating from the anchor", which frames 0 as the free option.
        Both are measured suspects for the manufactured bullish lean; see
        :data:`_ANCHOR_SUPPLEMENT_TWO_SIDED`. Default ``False`` keeps the
        string byte-identical to what the ``original`` and ``symloc`` runs
        were produced with, so those stay reproducible.
    """
    schema = AnchoredForecastOutput.prompt_schema_json()
    role = (
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical developments, and historical "
        "price dynamics. Oil markets move down as readily as up, and you are "
        "equally attentive to evidence in either direction."
        if two_sided
        else (
            "You are an expert WTI crude oil market analyst. You produce calibrated "
            "probabilistic price forecasts for WTI crude oil futures, grounded in "
            "supply/demand fundamentals, geopolitical risk, and historical price dynamics."
        )
    )
    instruction = (
        "## Role\n\n" + role + "\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels the anchor is expressed at\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n"
        "- `anchor`: a statistical baseline per horizon — see below\n\n"
        "Rules:\n"
        "1. Produce one signal pair for each horizon listed in `horizons`.\n"
        "2. `signal_loc` must be in [-1, 1] and `signal_width` must be in [0, 1] — both are "
        "enforced bounds, not suggestions.\n"
        + (
            "3. Each horizon's `rationale` is REQUIRED, not optional — it must contain the "
            "two-sided evidence check described below, including when a signal is 0.\n"
            "4. List the decisive cited evidence (OPEC+ decisions in either direction, supply "
            "disruptions or additions, inventory builds or draws, demand strength or weakness) "
            "in `key_signals` — this is compared against realised outcomes later.\n"
            if two_sided
            else (
                "3. Each horizon's `rationale` is REQUIRED, not optional — even when both signals are "
                "0, state what you checked and why it didn't warrant deviating from the anchor.\n"
                "4. List the decisive cited evidence (specific OPEC+ decisions, geopolitical events, "
                "inventory reports) in `key_signals` — this is compared against realised outcomes later.\n"
            )
        )
        + "5. When tools are enabled, conclude with `set_model_response` to return the structured output.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        "Omit any field not shown above."
    )
    if not include_search_guidance:
        return instruction
    return (
        instruction + "\n\n"
        "## Analysis discipline\n\n"
        "When context retrieval is available, call ``search_web`` to gather market "
        "intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended queries (call ``search_web`` once per topic):\n"
        '- ``search_web(query="WTI crude oil price trend and OPEC+ supply decisions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="Persian Gulf geopolitical risk shipping lane disruptions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="US Strategic Petroleum Reserve policy and global demand outlook", cutoff_date=<as_of>)``\n\n'
        "Document your key assumptions (OPEC+ policy, shipping lane risk, inventory "
        "levels, macro demand) in the `rationale` fields of your forecast output."
    )


# NOTE: _WTI_ANCHORED_ANALYST_INSTRUCTION[_CACHED_NEWS] are assigned further
# below, once AnchoredForecastOutput (a local class) is defined —
# _build_wti_anchored_instruction reads its schema and cannot run at module
# top before that class exists.

# ---------------------------------------------------------------------------
# Context retrieval instruction (sub-agent)
# ---------------------------------------------------------------------------

_WTI_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""

# ---------------------------------------------------------------------------
# Skills supplement (appended to instruction when skills are attached)
# ---------------------------------------------------------------------------

_CODE_EXEC_SKILLS_SUPPLEMENT = """

## Skills

You have access to two forecasting skills via the SkillToolset. All data
available to code execution comes from the JSON payload in your context —
there are no disk files to read.

**Recommended invocation order:**

1. `statistical-analysis` — run first. Provides diagnostic code patterns
   for interrogating the price series you have been given: vol regime
   classification, anomaly detection, and adaptive trend-window selection.
   The output of Pattern 3 (trend window) is the input to the projection
   skill below.

2. `trend-projection` — run second. Provides code patterns for fitting a
   linear trend on the window chosen above, projecting point forecasts to
   each horizon, and calibrating 80% prediction interval widths.

**To use a skill:**
1. Call `list_skills` to see available skill names and descriptions.
2. Call `load_skill(<name>)` to read the skill's full instructions.
3. Call `load_skill_resource(<skill_name>, <file_path>)` to load a
   reference file (e.g. `references/wti_benchmarks.json`).

These skills have NO scripts. Do not call `run_skill_script`.\
"""

# ---------------------------------------------------------------------------
# Forecast tool supplement (appended to instruction when the forecast tool is attached)
# ---------------------------------------------------------------------------

_FORECAST_TOOL_SUPPLEMENT = f"""

## Statistical forecast tool

You have access to `run_forecast`, a conventional statistical baseline
(AutoARIMA) you can call directly. Unlike open-ended code, this tool has a fixed,
auditable interface and returns a structured forecast you can reason from.

Call it ONCE before producing your forecast, with:
- `series_id`: "{WTI_SERIES_ID}"
- `cutoff_date`: the `as_of` date from the payload (YYYY-MM-DD). This is the
  information cutoff — the model uses only data on or before it.
- `horizons`: the `horizons` list from the payload.
- `frequency`: "B" (WTI trades on business days).

The tool returns JSON with point forecasts and 80%/90% prediction intervals per
horizon. Treat it as a disciplined statistical anchor: combine it with the
market context from the search sub-agent. You may adjust away from the baseline
when fundamentals or geopolitical risk justify it — document your reasoning in
the `rationale` fields.\
"""

# ---------------------------------------------------------------------------
# Anchor supplement (appended to instruction for the anchored variant)
# ---------------------------------------------------------------------------

_ANCHOR_SUPPLEMENT = """

## Statistical anchor

Your prompt payload includes `anchor` — an independently computed AutoARIMA
statistical baseline for each horizon (point forecast, quantile grid, and
`half_width`, half of its 90% interval). It is already computed; you do not
call any tool to produce it.

For each horizon, output two bounded numbers instead of a raw point forecast
and quantiles:
- `signal_loc` in [-1, +1]: direction/strength of believed drift away from the
  anchor's point forecast, as a fraction of the anchor's `half_width`. 0 means
  you trust the anchor's point forecast exactly.
- `signal_width` in [0, +1]: how much *more* uncertain this situation is than
  the anchor's own interval suggests. 0 means you trust the anchor's width
  exactly — there is no negative half, so narrowing below the anchor's own
  width is not an expressible output.

Document the concrete evidence (news, fundamentals, geopolitical risk) that
justifies any nonzero signal in `rationale`. If you have no such evidence,
leave both signals at 0 — but still write `rationale`: state what you
checked and that it revealed nothing beyond the anchor's own expectation.
A blank rationale is never acceptable, including in the zero-signal case.\
"""

# ---------------------------------------------------------------------------
# Symmetric-location supplement (experimental alternative to _ANCHOR_SUPPLEMENT)
# ---------------------------------------------------------------------------
#
# Measured problem this exists to test. Pairing the anchored and free-form
# agents on identical (origin, horizon, cached news) cells shows the anchored
# schema does not shrink the location signal symmetrically — it *rectifies* it.
# Where the free-form agent forecast below the anchor, the anchored agent
# mostly did not follow (gemini-3.1-flash-lite-preview: of 14 such cells, 1
# negative / 5 zero / 8 positive; gemini-3.5-flash: of 21, 9 / 2 / 10). Fitted
# pass-through ratios differ by side rather than matching as pure shrinkage
# would predict: +0.55 up vs -0.37 down (preview), +1.15 vs -0.08 (3.5-flash).
# The two agents agree on ranking (Spearman +0.64 / +0.62, p<0.001), so this is
# not disagreement about the market — the bottom half of the signal is being
# lost in expression.
#
# Hypothesis under test: in _ANCHOR_SUPPLEMENT the two bounded signals are
# adjacent bullets, and the `signal_width` bullet spends a full clause on
# "there is no negative half", while `signal_loc`'s negative half is only
# implied by the interval notation. The non-negativity framing may bleed onto
# the neighbouring signal. Consistent with the weaker, more prompt-suggestible
# model rectifying harder (2% negatives vs 24%).
#
# The change is deliberately confined to *framing*: the schema, bounds, and
# reconstruction arithmetic are identical, so any difference in the paired
# ratios is attributable to wording. Kept as a separate constant rather than
# an edit to _ANCHOR_SUPPLEMENT so the original run stays reproducible and the
# two can be compared directly.

_ANCHOR_SUPPLEMENT_SYMMETRIC_LOC = """

## Statistical anchor

Your prompt payload includes `anchor` — an independently computed AutoARIMA
statistical baseline for each horizon (point forecast, quantile grid, and
`half_width`, half of its 90% interval). It is already computed; you do not
call any tool to produce it.

### Signal 1 of 2 — `signal_loc`, where the price lands

`signal_loc` in [-1, +1]: how far you believe the true price will land away
from the anchor's point forecast, as a fraction of the anchor's `half_width`.

This signal is **symmetric — both directions are equally expressible and
equally expected**:
- **Negative** (-1 to 0) means you believe the true price will come in
  **BELOW** the anchor's point forecast. Use it whenever the evidence points
  down — bearish inventories, demand destruction, an OPEC+ supply increase,
  an easing risk premium.
- **Positive** (0 to +1) means you believe it will come in **ABOVE** the
  anchor's point forecast.
- **0** means you trust the anchor's point forecast exactly.

The anchor is a mechanical time-series extrapolation with no knowledge of
current events. It is as likely to sit too high as too low. If your reading of
the evidence is bearish relative to the anchor, report a negative `signal_loc`
of the same magnitude you would have reported for equally strong bullish
evidence. Do not treat 0 as a floor.

### Signal 2 of 2 — `signal_width`, how wide the interval is

`signal_width` in [0, +1]: how much *more* uncertain this situation is than
the anchor's own interval suggests. 0 means you trust the anchor's width
exactly. This second signal is about interval width only and is unrelated to
the direction you chose above; its range is one-sided for reasons specific to
interval calibration, which say nothing about `signal_loc`'s range.

### Both signals

Document the concrete evidence (news, fundamentals, geopolitical risk) that
justifies any nonzero signal in `rationale`. If you have no such evidence,
leave both signals at 0 — but still write `rationale`: state what you
checked and that it revealed nothing beyond the anchor's own expectation.
A blank rationale is never acceptable, including in the zero-signal case.\
"""

# ---------------------------------------------------------------------------
# Two-sided supplement (third anchor-prompt variant)
# ---------------------------------------------------------------------------
#
# What the first two variants left on the table. Converting the free-form
# cached-news runs into signal units (the "reversed contract" analysis) put the
# agent's own unanchored view on the same axis as `signal_loc`, over 42 paired
# (origin, horizon) cells on energy_oil_eval. The two models turn out to fail
# differently, and neither _ANCHOR_SUPPLEMENT nor _ANCHOR_SUPPLEMENT_SYMMETRIC_LOC
# closes the gap:
#
#                      own view (free-form)    anchored (original prompt)
#   preview   mean            0.233                    0.198
#             sd              0.443                    0.180   <- range crushed 2.5x
#             % negative       33%                       2%
#   3.5-flash mean           -0.006                    0.133   <- lean manufactured
#             sd              0.288                    0.216
#             % negative       50%                      24%
#
# So preview keeps its (genuine, in this window) bullish mean but loses the
# spread that would let it ever cross zero, while 3.5-flash's own view is
# centred at zero and the anchored format adds ~+0.13 out of nothing. The
# reported regression intercept is not an independent third effect: it is
# mean_anchored - slope * mean_freeform, which reproduces +0.142 / +0.136 to
# three decimals.
#
# Three suspects this variant addresses at once, all of which survived the
# symmetric-loc edit:
#
#   1. Directionally loaded vocabulary. Every evidence category named in the
#      prompt is one that pushes price UP -- "geopolitical risk" above all,
#      which in oil is definitionally additive. The observed rationales echo
#      it back ("maintaining a very slight positive bias for potential
#      lingering geopolitical risk premiums"). Handled here and via
#      `two_sided=True` on _build_wti_anchored_instruction.
#   2. Asymmetric burden of proof. Both earlier supplements ask for evidence
#      "that justifies any nonzero signal" and offer 0 as the fallback when
#      none is found, making 0 free and any move costly. That predicts
#      shrinkage toward 0, which is what preview shows.
#   3. Exhortation is not enough for the weaker model. _ANCHOR_SUPPLEMENT_SYMMETRIC_LOC
#      already told the agent negatives were equally expected; preview's
#      *prose* stayed non-bearish on 14/14 cells where its own free-form run
#      was bearish, so the view was gone before any number was chosen. This
#      variant therefore replaces exhortation with a forcing function: the
#      bearish case must be written out, first, before a number is picked.
#
# Deliberately a single prompt for both models rather than one tuned per model
# -- a per-model prompt would confound prompt with model and stop the arms
# being comparable.
#
# Pre-registered, label-free success criteria (no realised prices involved),
# measured against the same model's free-form run on the same cells:
#   - sd(anchored) / sd(free-form) -> 1.0   (now 0.41 preview, 0.75 3.5-flash)
#   - mean(anchored) - mean(free-form) -> 0 (now -0.04 preview, +0.14 3.5-flash)
# CRPS/coverage stay sealed until those move, so prompt iteration cannot
# overfit the eval window.

_ANCHOR_SUPPLEMENT_TWO_SIDED = """

## Reference forecast

Your prompt payload includes `anchor` — an independently computed AutoARIMA
extrapolation for each horizon (point forecast, quantile grid, and
`half_width`, half of its 90% interval). It is already computed; you do not
call any tool to produce it.

It is a mechanical extrapolation of past prices. It has no knowledge of current
events and holds no view about them. It is exactly as likely to sit **too high**
as **too low**. Your job is to say which of those it is — not to decide how far
to depart from a number that is presumed correct.

### Step 1 — the two-sided evidence check (write this before choosing numbers)

Each horizon's `rationale` must state, in this order:

1. **The strongest case that the price lands BELOW the extrapolation.** Name the
   most bearish concrete item in your evidence — oversupply, inventory builds,
   demand weakness, an OPEC+ output increase, a risk premium that is fading,
   macro deterioration, a resolved disruption. If you truly find none, say "no
   bearish evidence" — but look first.
2. **The strongest case that it lands ABOVE.** Name the most bullish concrete
   item — supply disruption, an OPEC+ cut, inventory draws, demand strength, a
   rising risk premium.
3. **Which case is stronger, and by how much.** That comparison is what your
   number encodes.

A rationale that argues only one side is incomplete. Naming a bearish factor and
then treating it only as a limit on the upside does **not** count as step 1 —
if the bearish case is real, it moves the price down, not merely less up.

### Step 2 — `signal_loc` in [-1, +1]: where the price lands

How far from the extrapolation's point forecast you believe the price will
actually land, as a fraction of `half_width`.

- **Negative** (−1 to 0) — below the extrapolation. Use it whenever step 1 came
  out stronger than step 2.
- **Positive** (0 to +1) — above it, when step 2 came out stronger.
- **0** — a substantive claim, not a safe default: it says the two sides cancel
  almost exactly. Justify it exactly as you would justify any other value.

For equally strong evidence, a bearish reading must produce a negative number of
the same magnitude a bullish reading would have produced. Nothing about oil
markets makes the upward case more likely to be the true one.

### Step 3 — `signal_width` in [0, +1]: how wide the interval is

How much *more* uncertain this situation is than the extrapolation's own
interval suggests. 0 means its width already fits. This signal's range is
one-sided for reasons specific to interval calibration; that says nothing about
`signal_loc`, which is fully two-sided.\
"""

# ---------------------------------------------------------------------------
# Cached-news supplement (appended instead of live search_web guidance when a
# NewsCacheSource is supplied — see build_wti_anchored_config)
# ---------------------------------------------------------------------------

_CACHED_NEWS_SUPPLEMENT = """

## Market news (pre-cached)

Your prompt payload includes `news_briefing` — a market intelligence summary
that was already retrieved with a hard cutoff at this origin's `as_of` date.
There is no `search_web` tool in this configuration; read `news_briefing`
directly and use it as your market/news context. Do not use your own
background knowledge to supplement it — if `news_briefing` seems thin, note
the gap in your rationale rather than filling it from memory.\
"""

# ---------------------------------------------------------------------------
# Skill directories
# ---------------------------------------------------------------------------

_SKILLS_ROOT = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


def compress_history(df: pd.DataFrame) -> str:
    """Compress WTI daily history to stay within context limits.

    Returns daily bars for the most recent 6 months and weekly averages for
    older history.  The CSV header is ``date,close``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``timestamp`` and ``value``.

    Returns
    -------
    str
        CSV string with header ``date,close``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=6)

    recent = df[df["timestamp"] >= cutoff].copy()
    old = df[df["timestamp"] < cutoff].copy()

    rows: list[str] = ["date,close"]

    if not old.empty:
        old_indexed = old.set_index("timestamp")["value"]
        weekly: pd.Series = old_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in recent.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_wti_payload(*, task: ForecastingTask, context: ForecastContext) -> dict[str, Any]:
    """Build the base WTI forecast payload shared by all prompt builders.

    Parameters
    ----------
    task : ForecastingTask
        The forecasting task — supplies ``task_id``, ``horizons``.
    context : ForecastContext
        The information state at forecast time.

    Returns
    -------
    dict[str, Any]
        Payload with task metadata, compressed history, and the standard
        quantile grid the agent must populate. Callers that need additional
        keys (e.g. an injected statistical anchor) may add them before
        serialising.
    """
    df = context.get_series(task.target_series_id)
    compressed = compress_history(df)

    last_row = df.iloc[-1]
    last_close = float(last_row["value"])
    last_date = str(pd.Timestamp(last_row["timestamp"]).date())
    trailing_252 = df["value"].tail(252)

    return {
        "task": task.task_id,
        "as_of": str(context.as_of)[:10],
        "horizons": list(task.horizons),
        "standard_quantiles": list(STANDARD_QUANTILES),
        "target_summary": {
            "last_close_usd_bbl": last_close,
            "last_date": last_date,
            "n_trading_days": int(len(df)),
            "52w_high": float(trailing_252.max()),
            "52w_low": float(trailing_252.min()),
        },
        "target_history_csv": compressed,
    }


class WtiPriceForecastPromptBuilder(BaseModel):
    """Prompt builder for WTI crude oil price forecasting tasks.

    Produces a structured JSON payload for the analyst agent containing the
    task specification, compressed price history, and a data summary.
    The payload includes ``standard_quantiles`` explicitly so the agent knows
    the exact grid it must produce.

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol (structural typing — no explicit inheritance required).

    Attributes
    ----------
    news_source : NewsCacheSource or None
        When supplied, a pre-cached news briefing for ``context.as_of`` is
        looked up and added to the payload as ``news_briefing`` instead of
        the agent calling ``search_web`` live — see
        :func:`build_wti_tool_config`'s ``news_source`` parameter, which must
        be given the same source so the config drops the live search tool to
        match. Mirrors :class:`AnchoredWtiPromptBuilder`'s parameter of the
        same name. ``None`` (default) leaves the payload exactly as before
        this parameter existed.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    news_source: NewsCacheSource | None = None

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise the task, context, and (optionally) cached news into a JSON string.

        Parameters
        ----------
        task : ForecastingTask
            The forecasting task — supplies ``task_id``, ``horizons``.
        context : ForecastContext
            The information state at forecast time.

        Returns
        -------
        str
            JSON-serialised payload with task metadata, compressed history, and
            the standard quantile grid the agent must populate, plus a
            ``news_briefing`` key when ``news_source`` is set.
        """
        payload = _build_wti_payload(task=task, context=context)
        if self.news_source is not None:
            payload["news_briefing"] = self.news_source.get(as_of=context.as_of)
        return json.dumps(payload, indent=2)


class AnchoredHorizonSignal(BaseModel):
    """Bounded location/scale signal for one horizon, relative to the anchor.

    Attributes
    ----------
    horizon : int
        Forecast horizon step (>= 1) corresponding to one entry of
        :attr:`~aieng.forecasting.evaluation.task.ForecastingTask.horizons`.
    signal_loc : float
        Direction/strength of believed drift away from the anchor's point
        forecast, as a fraction of the anchor's half-width. Bounded to
        ``[-1, +1]``; ``0`` means trust the anchor's point forecast exactly.
    signal_width : float
        How much *more* uncertain the situation is than the anchor's own
        interval suggests. Bounded to ``[0, +1]`` — there is no negative
        half, so narrowing below the anchor's own width is not a
        representable output.
    rationale : str
        Required horizon-specific explanation, propagated to
        ``Prediction.metadata["horizon_rationale"]``. Mandatory even when both
        signals are 0 — the null case ("checked available context, nothing
        beyond the anchor's own expectation") is exactly the record worth
        keeping for later audit, and an optional field tends to go blank
        precisely there.
    """

    model_config = {"extra": "ignore"}

    horizon: int = Field(ge=1, description="Forecast horizon step from the task, e.g. 1 for one period ahead.")
    signal_loc: float = Field(
        ge=-1.0,
        le=1.0,
        description="Direction/strength of drift from the anchor's point forecast, as a fraction of half_width.",
    )
    signal_width: float = Field(
        ge=0.0,
        le=1.0,
        description="Extra uncertainty beyond the anchor's own width. 0 = trust the anchor's width exactly.",
    )
    rationale: str = Field(
        min_length=1,
        description=(
            "Required explanation for this horizon's signal_loc/signal_width. Even when both are "
            "0, state what was checked and why it didn't warrant deviating from the anchor."
        ),
    )


class AnchoredForecastOutput(AgentForecastOutput):
    """Agent output for the anchored WTI forecasting variant.

    Instead of a free-form point forecast and quantile grid, the agent
    supplies two bounded signals per horizon relative to a statistical
    anchor it is shown in the prompt payload (see
    :class:`AnchoredWtiPromptBuilder`). Reconstruction into a final point
    forecast and quantile grid requires the anchor lookup table, which this
    schema has no access to — that happens in
    :class:`~energy_oil_forecasting.analyst_agent.anchored_predictor.AnchoredAgentPredictor.predict`,
    not in :meth:`to_predictions`.

    Attributes
    ----------
    signals : list[AnchoredHorizonSignal]
        One bounded signal pair per requested task horizon.
    rationale : str
        Optional overall explanation propagated to
        ``Prediction.metadata["rationale"]`` when non-empty.
    key_signals : list[str]
        Optional list of decisive cited evidence (e.g. specific OPEC+
        decisions, geopolitical events, inventory reports), mirroring
        :class:`~aieng.forecasting.methods.agentic.outputs.CategoricalAgentForecastOutput`'s
        field of the same name. Propagated to
        ``Prediction.metadata["key_signals"]`` when non-empty — a structured,
        machine-checkable companion to the free-text ``rationale`` fields.
    """

    modality: ClassVar[Literal["continuous", "discrete", "categorical"]] = "continuous"

    model_config = {"extra": "ignore"}

    signals: list[AnchoredHorizonSignal] = Field(
        description="One bounded signal pair for each requested task horizon.",
    )
    rationale: str = Field(
        default="", description="Optional overall explanation for the forecast; omit when not needed."
    )
    key_signals: list[str] = Field(default_factory=list, description="Key signals supporting the estimate.")

    @model_validator(mode="after")
    def _signal_horizons_are_unique(self) -> "AnchoredForecastOutput":
        """Reject empty or duplicate horizon signals before conversion."""
        if not self.signals:
            raise ValueError("signals must contain at least one horizon signal.")
        seen: set[int] = set()
        duplicates: list[int] = []
        for signal in self.signals:
            if signal.horizon in seen:
                duplicates.append(signal.horizon)
            seen.add(signal.horizon)

        if duplicates:
            raise ValueError(f"Duplicate signal horizons are not allowed: {duplicates}")
        return self

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings.

        Returns
        -------
        str
            Indented JSON string showing the exact structure the agent must
            pass to ``set_model_response``.
        """
        template: dict[str, object] = {
            "signals": [
                {
                    "horizon": "<integer — one entry per horizon from the task>",
                    "signal_loc": "<float in [-1, 1]>",
                    "signal_width": "<float in [0, 1]>",
                    "rationale": "<string>",
                }
            ],
            "rationale": "<string, optional overall explanation>",
            "key_signals": ["<signal 1>", "<signal 2>"],
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Not supported — reconstruction needs the anchor lookup table.

        Raises
        ------
        NotImplementedError
            Always. Use
            :class:`~energy_oil_forecasting.analyst_agent.anchored_predictor.AnchoredAgentPredictor`,
            whose overridden ``predict()`` has access to the ``AnchorSource``
            needed to turn these bounded signals into a final forecast.
        """
        raise NotImplementedError(
            "AnchoredForecastOutput must be converted via AnchoredAgentPredictor.predict(), "
            "which has access to the anchor lookup table needed for reconstruction."
        )


class AnchoredWtiPromptBuilder(BaseModel):
    """Prompt builder that injects a precomputed statistical anchor.

    Produces the same base payload as :class:`WtiPriceForecastPromptBuilder`
    plus an ``anchor`` key giving the agent, for each horizon, the
    independently-computed AutoARIMA point forecast, quantile grid, and
    half-width to deviate from via bounded signals (see
    :class:`AnchoredForecastOutput`).

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol (structural typing — no explicit inheritance required).

    Attributes
    ----------
    anchor_source : AnchorSource
        Precomputed anchor lookup table to read from. Loaded once by the
        caller (e.g. via :meth:`AnchorSource.from_spec_id`) and shared with
        the :class:`~energy_oil_forecasting.analyst_agent.anchored_predictor.AnchoredAgentPredictor`
        that reconstructs the final forecast from the same anchors.
    news_source : NewsCacheSource or None
        When supplied, a pre-cached news briefing for ``context.as_of`` is
        looked up and added to the payload as ``news_briefing`` instead of
        the agent calling ``search_web`` live — see
        :func:`build_wti_anchored_config`'s ``news_source`` parameter, which
        must be given the same source so the config drops the live search
        tool to match. ``None`` (default) leaves the payload exactly as
        before this parameter existed.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    anchor_source: AnchorSource
    news_source: NewsCacheSource | None = None

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise the task, context, anchor, and (optionally) cached news into a JSON string.

        Parameters
        ----------
        task : ForecastingTask
            The forecasting task — supplies ``task_id``, ``horizons``.
        context : ForecastContext
            The information state at forecast time.

        Returns
        -------
        str
            JSON-serialised payload identical to
            :class:`WtiPriceForecastPromptBuilder`'s, plus an ``anchor`` key
            keyed by horizon, plus a ``news_briefing`` key when
            ``news_source`` is set.
        """
        payload = _build_wti_payload(task=task, context=context)
        anchor_by_horizon: dict[str, Any] = {}
        for horizon in task.horizons:
            entry = self.anchor_source.get(as_of=context.as_of, horizon=horizon)
            anchor_by_horizon[str(horizon)] = {
                "point_forecast": entry.point_forecast,
                "half_width": entry.half_width,
                "quantiles": {str(q): v for q, v in entry.quantiles.items()},
            }
        payload["anchor"] = anchor_by_horizon
        if self.news_source is not None:
            payload["news_briefing"] = self.news_source.get(as_of=context.as_of)
        return json.dumps(payload, indent=2)


# AnchoredForecastOutput now exists, so the instruction that embeds its schema
# can be built (see the NOTE left where _WTI_ANALYST_INSTRUCTION is assigned).
_WTI_ANCHORED_ANALYST_INSTRUCTION = _build_wti_anchored_instruction()
_WTI_ANCHORED_ANALYST_INSTRUCTION_CACHED_NEWS = _build_wti_anchored_instruction(include_search_guidance=False)
_WTI_ANCHORED_ANALYST_INSTRUCTION_TWO_SIDED = _build_wti_anchored_instruction(two_sided=True)
_WTI_ANCHORED_ANALYST_INSTRUCTION_CACHED_NEWS_TWO_SIDED = _build_wti_anchored_instruction(
    include_search_guidance=False, two_sided=True
)

# anchor-prompt variant -> (base instruction, cached-news instruction, supplement,
# config-name suffix). The suffix is what separates each variant's persisted
# predictions, so existing files keep resolving: "original" must stay "".
ANCHOR_PROMPT_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "original": (
        _WTI_ANCHORED_ANALYST_INSTRUCTION,
        _WTI_ANCHORED_ANALYST_INSTRUCTION_CACHED_NEWS,
        _ANCHOR_SUPPLEMENT,
        "",
    ),
    "symloc": (
        _WTI_ANCHORED_ANALYST_INSTRUCTION,
        _WTI_ANCHORED_ANALYST_INSTRUCTION_CACHED_NEWS,
        _ANCHOR_SUPPLEMENT_SYMMETRIC_LOC,
        "_symloc",
    ),
    "twosided": (
        _WTI_ANCHORED_ANALYST_INSTRUCTION_TWO_SIDED,
        _WTI_ANCHORED_ANALYST_INSTRUCTION_CACHED_NEWS_TWO_SIDED,
        _ANCHOR_SUPPLEMENT_TWO_SIDED,
        "_twosided",
    ),
}

# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_wti_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build an :class:`AgentConfig` with no tools.

    The agent reasons purely from the price history in the prompt payload.
    Useful as a low-cost baseline or starting point when comparing capability
    levels.

    Parameters
    ----------
    model : str
        Gemini model identifier.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_basic",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
    )


def build_wti_multitask_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """News-grounded config for the one-agent-three-tasks demo (NB3).

    Uses a task-agnostic analyst instruction; the task schema is supplied in
    the user prompt payload via :class:`~energy_oil_forecasting.tasks.WtiMultitaskPromptBuilder`.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
    """
    return AgentConfig(
        name="wti_analyst_multitask",
        model=model,
        instruction=_WTI_MULTITASK_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
    *,
    news_source: NewsCacheSource | None = None,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with bounded Google Search.

    Wires a :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
    sub-agent that enforces a temporal cutoff on every search call, preventing
    future information from contaminating historical backtests. An
    independent verifier call audits each search result against the cutoff
    before it reaches the analyst (see :class:`ContextRetrievalConfig`).

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
        Unused when ``news_source`` is supplied.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots. Unused when ``news_source`` is supplied.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel. Unused when
        ``news_source`` is supplied.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
        Unused when ``news_source`` is supplied.
    news_source : NewsCacheSource or None
        When supplied, context retrieval is disabled entirely and the agent
        reads a pre-cached briefing injected by
        :class:`WtiPriceForecastPromptBuilder` (which must be given the same
        ``news_source``) instead of calling ``search_web`` live. Exactly
        mirrors :func:`build_wti_anchored_config`'s parameter of the same
        name, so the free-form and anchored variants can be compared on
        byte-identical news.

    Returns
    -------
    AgentConfig
    """
    if news_source is not None:
        return AgentConfig(
            name="wti_analyst_news_cached",
            model=model,
            instruction=_WTI_ANALYST_INSTRUCTION_CACHED_NEWS + _CACHED_NEWS_SUPPLEMENT,
        )

    return AgentConfig(
        name="wti_analyst_news",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_code_exec_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with E2B code execution and forecasting skills.

    Combines bounded Google Search (temporal cutoff enforced) with E2B sandbox
    code execution and two forecasting skills:

    - ``statistical-analysis``: diagnostic patterns for the payload data
      (vol regime, anomaly detection, adaptive trend window).
    - ``trend-projection``: linear trend fit, CI calibration, and plausibility
      guard using the window determined by statistical-analysis.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    max_output_tokens : int, default=16_384
        Maximum tokens per model response.  The default is set well above
        LiteLLM's OpenAI-compatible endpoint default of 4096, which is not
        enough for Claude to write a complete ``run_code`` Python script in a
        single function call — causing repeated retries with empty arguments.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_code",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION + _CODE_EXEC_SKILLS_SUPPLEMENT,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        skills_dirs=[
            _SKILLS_ROOT / "statistical-analysis",
            _SKILLS_ROOT / "trend-projection",
        ],
    )


def build_wti_tool_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    num_samples: int = 200,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with a conventional statistical forecast tool.

    This is the fourth analyst capability level. It combines bounded Google
    Search (temporal cutoff enforced) with a
    :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
    that runs AutoARIMA on the WTI series. In contrast to
    :func:`build_wti_code_exec_config` — which gives the agent open-ended code
    execution — this path exposes a rigid, pre-specified tool, trading
    flexibility for control and reproducibility.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    data_service : DataService or None
        Pre-populated data service with the WTI series registered. When
        ``None``, one is constructed via
        :func:`~energy_oil_forecasting.data.build_wti_service` (cache-backed).
        Series data is read by the tool but never enters the LLM context.
    num_samples : int, default=200
        Monte Carlo sample count for AutoARIMA. Kept modest to bound agent
        latency, since AutoARIMA can be slow per origin.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    service = data_service if data_service is not None else build_wti_service()
    forecast_tool = ForecastTool(service, predictor=DartsAutoARIMAPredictor(num_samples=num_samples))

    return AgentConfig(
        name="wti_analyst_tool",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION + _FORECAST_TOOL_SUPPLEMENT,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        function_tools=[forecast_tool.as_function_tool()],
    )


def build_wti_anchored_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    news_source: NewsCacheSource | None = None,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
    anchor_prompt: str = "original",
) -> AgentConfig:
    """Build an :class:`AgentConfig` for the anchored WTI variant.

    Same news-grounding shape as :func:`build_wti_tool_config` (bounded
    Google Search with temporal cutoff enforcement), but with **no**
    ``function_tools`` — the statistical anchor is already injected into the
    prompt payload by :class:`AnchoredWtiPromptBuilder`, so a ``run_forecast``
    tool would let the agent compute a second, inconsistent anchor instead of
    reasoning from the one the harness holds. ``build_wti_tool_config`` itself
    is unmodified; this is an additive sibling.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Ignored when
        ``news_source`` is supplied (context retrieval is disabled entirely).
    news_source : NewsCacheSource or None
        When supplied, ``context_retrieval`` is disabled and the agent is
        instructed to read the pre-cached ``news_briefing`` from the payload
        (added by :class:`AnchoredWtiPromptBuilder` when given the same
        ``news_source``) instead of calling ``search_web`` live. This trades
        news freshness for reproducibility: a live search can return a
        different result for the same origin on different runs, which
        contaminates mechanism-isolation comparisons the same way an
        unseeded anchor did — see
        ``planning-docs/anchor-externalization-interview-notes.md``.
        ``None`` (default) keeps live search, matching the original
        behavior before this parameter existed.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Unused when ``news_source`` is supplied.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel. Unused when
        ``news_source`` is supplied.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
        Unused when ``news_source`` is supplied.
    anchor_prompt : str, default="original"
        Which anchor-prompt framing to use — a key of
        :data:`ANCHOR_PROMPT_VARIANTS`. All three describe the *same* schema,
        bounds, and reconstruction arithmetic; only the wording differs, and
        each is renamed so its predictions persist to a separate file and the
        variants can be compared on identical origins.

        - ``"original"`` — :data:`_ANCHOR_SUPPLEMENT`, the shipped default.
        - ``"symloc"`` — :data:`_ANCHOR_SUPPLEMENT_SYMMETRIC_LOC`, which states
          ``signal_loc``'s negative half as explicitly as ``signal_width``'s
          absence of one.
        - ``"twosided"`` — :data:`_ANCHOR_SUPPLEMENT_TWO_SIDED`, which forces a
          written bearish case before a number is chosen and neutralises the
          directionally loaded vocabulary in the Role line and rules 3-4.

    Returns
    -------
    AgentConfig

    Raises
    ------
    KeyError
        If ``anchor_prompt`` is not a known variant.
    """
    try:
        base_instruction, cached_instruction, supplement, suffix = ANCHOR_PROMPT_VARIANTS[anchor_prompt]
    except KeyError as exc:
        raise KeyError(
            f"Unknown anchor_prompt {anchor_prompt!r}. Available: {sorted(ANCHOR_PROMPT_VARIANTS)}"
        ) from exc

    if news_source is not None:
        return AgentConfig(
            name=f"wti_analyst_anchored_cached_news{suffix}",
            model=model,
            instruction=cached_instruction + supplement + _CACHED_NEWS_SUPPLEMENT,
        )

    return AgentConfig(
        name=f"wti_analyst_anchored{suffix}",
        model=model,
        instruction=base_instruction + supplement,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_wti_agent_predictor(
    config: AgentConfig,
    *,
    news_source: NewsCacheSource | None = None,
) -> AgentPredictor:
    """Wrap an :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`WtiPriceForecastPromptBuilder` and
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`
    as the output schema.

    Parameters
    ----------
    config : AgentConfig
        Any of the configs produced by :func:`build_wti_basic_config`,
        :func:`build_wti_news_config`, or :func:`build_wti_code_exec_config`.
    news_source : NewsCacheSource or None
        Passed to the prompt builder so the agent sees a pre-cached briefing
        instead of calling ``search_web`` live. Must be the same source given
        to :func:`build_wti_news_config`, which drops the live search tool to
        match — otherwise the agent gets a briefing *and* a search tool, or a
        search-free config with no briefing.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiPriceForecastPromptBuilder(news_source=news_source),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_wti_basic_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
