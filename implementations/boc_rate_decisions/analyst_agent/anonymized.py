"""De-identified variant of the BoC analyst, for measuring recall contamination.

Why this exists
---------------
The canonical backtest runs over 120 fixed announcement dates from 2010 to
2024 — a window that sits entirely inside any current model's training data.
Bank of Canada decisions are discrete, widely reported events, so "what did
the Bank do in March 2022" is far more recallable than, say, a particular
day's WTI close. On a three-way categorical task scored with RPS, recall is
also far more exploitable than it would be on a continuous target.

The existing analyst instruction asks the model not to use that knowledge:

    Use ONLY information available on or before `as_of`. Do not use knowledge
    of what the Bank actually decided ... even if you remember it.

That is a prose-only constraint, and this repo has already established that
prose-only constraints do not bind — the adaptive WTI agent was told to call
the forecast tool before forecasting and fitted its own ARIMA in the sandbox
instead.

The approach here is not to ask harder. It is to **remove the information**:
:class:`AnonymizedPromptBuilder` emits the same predictive content with every
identifying handle stripped, so recall is not merely discouraged but
unavailable. Pairing it against the identified builder on the same origins
turns contamination into a measured quantity:

===========  ===========================================  =====================
arm          sees                                         isolates
===========  ===========================================  =====================
B (this)     macro state, undated, unattributed           reasoning only
C           macro state + real dates + "Bank of Canada"    ``C - B`` = recall
===========  ===========================================  =====================

Neither arm gets news, so neither is exposed to the *other* contamination
route — a 2026 web search for a 2022 decision returns articles stating the
outcome, which would put the answer directly in the context window.

Payload versions
----------------
**v2 (2026-08-24)** tightened the de-identification after a re-identification
probe measured what v1 actually leaked: asked to name the institution and the
year, the model got the institution wrong (15% named Canada, passing its bar)
but tracked the year closely — Spearman rho of +0.55 against the truth. Two
changes, both close to free in predictive content:

- the **meeting counter is gone**. ``n_meetings_observed`` ran 8, 9, 10 ... 127
  in lockstep with the origin: a clock with no predictive content of its own.
  ``counts`` was the same clock scaled, and the two explicit list lengths
  (``meetings_shown`` / ``window_meetings``) ramped 8 -> 40 across the early
  origins, which is the same clock again in a smaller font. Base rates — the part the instruction
  actually tells the agent to anchor on — are kept, now computed over a fixed
  trailing window so the denominator no longer moves, and rounded to 2 decimals
  so the original counts cannot be recovered by division.
- the **policy rate level is bucketed** into how far it sits above its
  effective floor. A level of 5.00 is unique to one period in this window;
  "well above the floor" is not. Distance to the floor is the part that carries
  real predictive weight (it bounds how far the bank can cut), and the recent
  *change* in the rate is supplied separately as ``rate_momentum``.

Predictions stored before 2026-08-24 (``data/recall_probe/``) were produced
against v1 and are not reproducible from this builder. Arm B is no longer half
of a matched contrast — it is a standalone testbed scored against the
deterministic logistic baseline — so v2 deliberately does not preserve
comparability with the stored arm C.

What is preserved
-----------------
Everything with predictive content: the four macro features, the current
policy rate level, the ordered sequence of past decisions, and the realised
base rates. What is removed: absolute dates, the institution's name and
country, the inflation-target figure, the currency, and the named bond
series. The decision history is kept as an ordered sequence with relative
offsets ("12 meetings ago") rather than calendar dates.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig
from pydantic import BaseModel

from aieng.forecasting.methods.agentic.outputs import CategoricalAgentForecastOutput

from ..data import (
    BOND_YIELD_2YR_SERIES_ID,
    CPI_SERIES_ID,
    TARGET_RATE_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
)
from ..predictors.logistic_baseline import build_feature_row
from .agent import LITE_MODEL


#: Bumped whenever the payload shape changes; see "Payload versions" above.
ANONYMIZED_PAYLOAD_VERSION = 2

#: Upper edges (in percentage points above the effective floor) of the bands
#: the policy rate is reported in, with the label each band carries.
_RATE_BANDS: tuple[tuple[float, str], ...] = (
    (0.25, "at its effective floor"),
    (1.25, "slightly above its floor"),
    (2.75, "moderately above its floor"),
    (float("inf"), "well above its floor"),
)

#: The lowest level the policy rate has ever been set to in this series. Used
#: as the floor the bands are measured from.
_EFFECTIVE_FLOOR_PCT = 0.25


def _policy_rate_band(rate_pct: float) -> str:
    """Coarse band for the policy rate, measured from its effective floor.

    The level itself is withheld: within a single bank's history a level is
    close to a timestamp, while the distance to the floor is what bounds how
    much room the bank has left to cut.
    """
    above_floor = float(rate_pct) - _EFFECTIVE_FLOOR_PCT
    for edge, label in _RATE_BANDS:
        if above_floor <= edge:
            return label
    return _RATE_BANDS[-1][1]


def _build_anonymized_instruction() -> str:
    """Build the de-identified instruction, embedding the live output schema.

    The schema block is generated from
    :class:`~aieng.forecasting.methods.agentic.outputs.CategoricalAgentForecastOutput`
    exactly as the identified instruction does, so the two arms cannot drift
    apart on output format — a format difference would show up as an arm
    difference and be misread as recall.
    """
    schema = CategoricalAgentForecastOutput.prompt_schema_json(labels=["cut", "hold", "hike"])
    return _ANONYMIZED_BODY + "## Output schema\n\n" + (
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
    ) + "```json\n" + schema + "\n```\n"


_ANONYMIZED_BODY = (
    "## Role\n\n"
    "You are a monetary-policy analyst. You will be shown the anonymised state "
    "of an unnamed central bank and asked for a calibrated probability "
    "distribution over what it does to its policy rate at its next scheduled "
    "decision: CUT (lower), HOLD (unchanged), or HIKE (raise).\n\n"
    "The institution, country, currency and calendar dates have been withheld "
    "deliberately. Do not guess which central bank this is, and do not condition "
    "on any guess — the evaluation depends on your answer coming from the "
    "supplied state alone. Reason from the numbers you are given.\n\n"
    "## Forecasting contract\n\n"
    "You will receive a JSON payload containing:\n"
    "- `policy_rate`: how far the policy rate currently sits above its "
    "effective floor, as a coarse band rather than a level, plus the ordered "
    "sequence of recent decisions\n"
    "- `meeting_outcomes`: the realised base rate of each outcome over the "
    "same trailing window of meetings\n"
    "- `macro_snapshot`: four leak-safe indicators as of the decision origin — "
    "inflation gap against the bank's stated target, unemployment momentum, "
    "the 2-year government yield minus the policy rate, and the trailing "
    "90-day change in the policy rate\n\n"
    "Rules:\n"
    "1. Assign one probability to each of `cut`, `hold`, and `hike` — a move of "
    "any size counts. The three must sum to 1.\n"
    "2. Report CALIBRATED probabilities, not your confidence in a point view: "
    "across many questions where you assign 0.7 to an outcome, that outcome "
    "should occur about 70% of the time. Anchor on the supplied base rates, "
    "then adjust.\n"
    "3. Cuts and hikes cluster into easing and tightening cycles; the rate "
    "momentum and decision history tell you whether you are in one. A 2-year "
    "yield well below the policy rate means the bond market is pricing cuts; "
    "well above means it is pricing hikes. Direct cut-to-hike reversals between "
    "adjacent meetings essentially never happen.\n"
    "4. Document your reasoning in `reasoning` and list the decisive inputs in "
    "`key_signals`.\n\n"
)

_ANONYMIZED_INSTRUCTION = _build_anonymized_instruction()


class AnonymizedPromptBuilder(BaseModel):
    """Emit the BoC payload with every identifying handle removed.

    Predictive content is preserved exactly; identity is not. See the module
    docstring for the full list of what is kept and what is stripped.
    """

    max_history: int = 40
    """How many past decisions to include, most recent last."""

    base_rate_window: int = 40
    """Meetings the base rates are computed over.

    Fixed rather than cumulative on purpose: a denominator that grows with the
    origin is a clock. A trailing window also puts the base rate closer to the
    current policy regime than an all-history average would.
    """

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise an anonymised payload for one origin.

        Raises
        ------
        ValueError
            If the task does not declare categories.
        """
        if task.categories is None:
            raise ValueError(f"{type(self).__name__} requires a categorical task with declared categories.")

        as_of = pd.Timestamp(context.as_of)
        rate_df = context.get_series(TARGET_RATE_SERIES_ID)
        yield_df = context.get_series(BOND_YIELD_2YR_SERIES_ID)
        cpi_df = context.get_series(CPI_SERIES_ID)
        unemployment_df = context.get_series(UNEMPLOYMENT_SERIES_ID)
        direction_df = context.get_series(task.target_series_id)

        features = build_feature_row(as_of, rate_df, yield_df, cpi_df, unemployment_df)

        labels_by_value = {category.value: category.label for category in task.categories}
        decisions = [labels_by_value[float(v)] for v in direction_df["value"] if float(v) in labels_by_value]
        recent = decisions[-self.max_history :]

        # Base rates over a fixed trailing window, rounded to 2 decimals. Both
        # choices exist to keep the meeting counter from leaking back in: a
        # cumulative denominator is a clock, and a 4-decimal rate divides back
        # into the raw counts.
        window = decisions[-self.base_rate_window :]
        base_rates = (
            {category.label: round(window.count(category.label) / len(window), 2) for category in task.categories}
            if window
            else None
        )

        payload: dict[str, Any] = {
            "task": {
                "question": (
                    "At this central bank's next scheduled policy decision, will it "
                    "CUT, HOLD, or HIKE its policy rate?"
                )
            },
            "policy_rate": {
                # A band, not a level: within one bank's history a level such
                # as 5.00 belongs to exactly one period.
                "level_relative_to_floor": _policy_rate_band(float(rate_df["value"].iloc[-1])),
                # Offsets in meetings, not dates: a dated change history would
                # re-identify the institution immediately.
                "recent_decisions_most_recent_last": recent,
            },
            "meeting_outcomes": {
                "base_rates_over_window": base_rates,
            },
            "macro_snapshot": (
                {k: round(float(v), 4) for k, v in features.items()}
                if features is not None
                else "insufficient history at this origin"
            ),
        }
        return json.dumps(payload, indent=2)


def build_boc_anonymized_config(model: str = LITE_MODEL) -> AgentConfig:
    """Config for the de-identified arm — no tools, no news, no dates.

    Deliberately mirrors :func:`..agent.build_boc_basic_config` in everything
    except the instruction and the payload, so that arm-vs-arm differences are
    attributable to the withheld identity and nothing else.
    """
    return AgentConfig(
        name="boc_analyst_anonymized",
        model=model,
        instruction=_ANONYMIZED_INSTRUCTION,
    )
