"""Bank of Canada policy analyst agent configuration and prompt builder.

Provides :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`
factories for the primary BoC 3-way rate-direction prediction task
(cut / hold / hike at the next fixed announcement date):

1. :func:`build_boc_basic_config` — the quantitative-only analyst: reasons
   from the policy-rate path, past meeting outcomes, and a leak-safe macro
   snapshot supplied in the prompt payload. No tools.
2. :func:`build_boc_news_config` — adds bounded Google Search via a
   :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
   sub-agent with strict temporal cutoffs.
3. :func:`build_boc_research_config` — deterministic document grounding from
   cached, cutoff-visible Bank communications, with no live search.

Also provides:

- :class:`BoCDecisionPromptBuilder` — Pydantic ``BaseModel`` that serialises
  the task, meeting calendar position, rate path, per-meeting decision
  history, macro snapshot, and optional research evidence into structured JSON.
- :func:`build_boc_agent_predictor` — convenience factory wiring a config to
  an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
  with the
  :class:`~aieng.forecasting.methods.agentic.outputs.CategoricalAgentForecastOutput`
  schema. The agent's ``reasoning`` / ``key_signals`` output fields are the
  hook for the LLM reasoning-alignment evaluation against the Bank's
  own published rationale.

The agent is direction-native: the compact binary rate-cut reference uses the
frequency baseline, logistic regression, and the binary LLMP recipe, and the
task-agnostic
:class:`~aieng.forecasting.methods.agentic.outputs.DiscreteAgentForecastOutput`
remains available in the core package for naturally binary problems.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    CategoricalAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import LITE_MODEL
from boc_rate_decisions.data import (
    BOND_YIELD_2YR_SERIES_ID,
    CPI_SERIES_ID,
    TARGET_RATE_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
)
from boc_rate_decisions.predictors.logistic_baseline import build_feature_row
from boc_rate_decisions.research import DEFAULT_RESEARCH_SOURCES, format_research_evidence
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------


def _build_boc_analyst_instruction() -> str:
    """Build the BoC analyst instruction, embedding the output schema from the class.

    Using a function instead of a static string ensures the ``## Output
    schema`` block is always in sync with ``CategoricalAgentForecastOutput``
    — no manual JSON to maintain.
    """
    schema = CategoricalAgentForecastOutput.prompt_schema_json(labels=["cut", "hold", "hike"])
    return (
        "## Role\n\n"
        "You are an expert Bank of Canada monetary-policy analyst. You produce a "
        "calibrated probability distribution over what the Bank does to its target "
        "for the overnight rate at a specific upcoming fixed announcement date — "
        "CUT (lower), HOLD (unchanged), or HIKE (raise) — grounded in the "
        "policy-rate path, the Bank's 2% CPI inflation target, labour-market and "
        "bond-market conditions, and the Bank's institutional behaviour "
        "(gradualism, data dependence, reluctance to surprise markets).\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier and question\n"
        "- `as_of`: the forecast origin date (YYYY-MM-DD) — your information cutoff\n"
        "- `announcement_date`: the fixed announcement date being predicted; the "
        "gap between `as_of` and this date is your forecast lead time\n"
        "- `policy_rate`: current target rate and the dated history of past rate "
        "changes\n"
        "- `meeting_outcomes`: per-meeting decision history (cut / hold / hike) "
        "with the realised base rates for each outcome\n"
        "- `macro_snapshot`: leak-safe indicators as of the origin (CPI inflation "
        "vs the 2% target, unemployment momentum, 2-year GoC yield vs the policy "
        "rate)\n"
        "- `research_evidence` (research-grounded variant only): dated, cutoff-filtered "
        "Bank documents with stable ids and source provenance\n\n"
        "Rules:\n"
        "1. Assign one probability to each of `cut`, `hold`, and `hike` — a move "
        "of any size counts. The three probabilities must sum to 1.\n"
        "2. Report CALIBRATED probabilities, not your confidence in a point view: "
        "across many questions where you assign 0.7 to an outcome, that outcome "
        "should occur about 70% of the time. Anchor on the historical base rates, "
        "then adjust.\n"
        "3. Cuts and hikes cluster into easing and tightening cycles; the macro "
        "snapshot tells you whether you are in one. The 2-year yield trading well "
        "below the policy rate means the bond market is pricing cuts; well above "
        "means it is pricing hikes. Direct cut-to-hike reversals between adjacent "
        "meetings essentially never happen, so the recent decision history should "
        "strongly shape which tail outcome is plausible.\n"
        "4. Use ONLY information available on or before `as_of`. Do not use "
        "knowledge of what the Bank actually decided on or after "
        "`announcement_date`, even if you remember it. If `search_web` returns a "
        "result beginning with `[SEARCH_VERIFICATION_FAILED]`, treat it as no "
        "verified news for that query — proceed on the other signals in your "
        "payload and note the gap, never filling it from your own background "
        "knowledge.\n"
        "5. Document your reasoning in `reasoning` and list the decisive inputs "
        "in `key_signals` — these are compared against the Bank's own published "
        "rationale by a downstream evaluator, so be specific.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching "
        "**exactly**:\n\n"
        "```json\n" + schema + "\n```\n"
    )


_BOC_ANALYST_INSTRUCTION = _build_boc_analyst_instruction()


# ---------------------------------------------------------------------------
# Context retrieval instruction (sub-agent) — seam for the report-grounded variant
# ---------------------------------------------------------------------------

_BOC_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a Canadian monetary-policy intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- Recent Bank of Canada communications: statements, speeches, Monetary Policy Reports
- Canadian CPI inflation prints and core-inflation measures vs the 2% target
- Canadian labour market: employment reports, unemployment rate, wage growth
- Market pricing of the upcoming decision (overnight index swaps, economist surveys)
- Macro shocks relevant to Canada: oil prices, exchange rate, US policy, trade

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
# Prompt builder
# ---------------------------------------------------------------------------


def _rate_change_history(rate_df: pd.DataFrame, max_changes: int = 40) -> list[dict[str, object]]:
    """Compress the daily step-wise policy rate into its change points.

    The daily series is constant between decisions, so the list of
    ``(date, new_rate)`` change points carries all of its information in a
    tiny fraction of the tokens.
    """
    values = rate_df["value"].astype(float)
    changed = values.diff().fillna(0.0) != 0.0
    changes = rate_df.loc[changed, ["timestamp", "value"]].tail(max_changes)
    return [
        {"date": str(pd.Timestamp(ts).date()), "new_target_rate_pct": float(v)}
        for ts, v in zip(changes["timestamp"], changes["value"])
    ]


class BoCDecisionPromptBuilder(BaseModel):
    """Prompt builder for the BoC 3-way rate-direction prediction task.

    Produces a structured JSON payload containing the question, the policy
    rate path (compressed to change points), the per-meeting decision history
    (serialised as ``cut`` / ``hold`` / ``hike`` labels with per-outcome base
    rates), and a leak-safe macro snapshot (shared with the logistic baseline
    via
    :func:`~boc_rate_decisions.predictors.logistic_baseline.build_feature_row`,
    so the agent and the conventional model see exactly the same indicators).

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol (structural typing — no explicit inheritance required).
    """

    document_sources: tuple[str, ...] = ()
    max_documents: int = 3
    max_chars_per_document: int = 6_000

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise the task and cutoff-filtered context into a JSON payload.

        Parameters
        ----------
        task : ForecastingTask
            The categorical rate-direction task — supplies ``task_id``,
            ``description``, the ordered ``categories`` mapping series values
            to labels, and the single-step horizon used to derive the
            announcement date.
        context : ForecastContext
            The information state at forecast time (cutoff-enforced).

        Returns
        -------
        str
            JSON-serialised payload for the analyst agent.

        Raises
        ------
        ValueError
            If the task does not declare categories or an observed outcome
            does not match any declared category value.
        """
        if task.categories is None:
            raise ValueError(f"{type(self).__name__} requires a categorical task with declared categories.")

        as_of = pd.Timestamp(context.as_of)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        announcement_date = as_of + offset * task.horizons[0]

        direction_df = context.get_series(task.target_series_id)
        rate_df = context.get_series(TARGET_RATE_SERIES_ID)
        yield_df = context.get_series(BOND_YIELD_2YR_SERIES_ID)
        cpi_df = context.get_series(CPI_SERIES_ID)
        unemployment_df = context.get_series(UNEMPLOYMENT_SERIES_ID)

        features = build_feature_row(as_of, rate_df, yield_df, cpi_df, unemployment_df)

        labels_by_value = {category.value: category.label for category in task.categories}
        outcomes: list[dict[str, object]] = []
        counts = {category.label: 0 for category in task.categories}
        for ts, value in zip(direction_df["timestamp"], direction_df["value"]):
            label = labels_by_value.get(float(value))
            if label is None:
                raise ValueError(
                    f"Observed outcome {float(value)} does not match any task category value "
                    f"({sorted(labels_by_value)})."
                )
            outcomes.append({"announcement_date": str(pd.Timestamp(ts).date()), "decision": label})
            counts[label] += 1

        n_meetings = len(outcomes)
        base_rates = {label: round(count / n_meetings, 4) for label, count in counts.items()} if n_meetings else None

        payload: dict[str, Any] = {
            "task": {"task_id": task.task_id, "question": task.description},
            "as_of": str(as_of.date()),
            "announcement_date": str(announcement_date.date()),
            "policy_rate": {
                "current_target_rate_pct": float(rate_df["value"].iloc[-1]),
                "rate_changes": _rate_change_history(rate_df),
            },
            "meeting_outcomes": {
                "history": outcomes,
                "n_meetings": n_meetings,
                "counts": counts,
                "historical_base_rates": base_rates,
            },
            "macro_snapshot": features if features is not None else "insufficient history at this origin",
        }
        if self.document_sources:
            payload["research_evidence"] = format_research_evidence(
                context,
                sources=self.document_sources,
                max_documents=self.max_documents,
                max_chars_per_document=self.max_chars_per_document,
            )
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_boc_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build the quantitative-only BoC analyst config (no tools).

    The agent reasons purely from the rate path, outcome history, and macro
    snapshot in the prompt payload — the same information set as the
    logistic baseline, making the comparison between conventional fitting
    and LLM reasoning clean.

    Parameters
    ----------
    model : str
        Model identifier for the analyst agent.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="boc_analyst_basic",
        model=model,
        instruction=_BOC_ANALYST_INSTRUCTION,
    )


def build_boc_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
) -> AgentConfig:
    """Build the news-grounded BoC analyst config (bounded Google Search).

    Wires a context-retrieval sub-agent that enforces a temporal cutoff on
    every search call. This factory is the seam for the deferred
    report-grounded variant: replacing the retrieval instruction (and later,
    the retrieval tool) with BoC press-release / MPR retrieval upgrades the
    agent without changing the forecasting contract.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so
        that Gemini handles Google Search even when the analyst uses a
        different provider.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="boc_analyst_news",
        model=model,
        instruction=_BOC_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_BOC_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
        ),
    )


def build_boc_research_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build the deterministic, cached-document-grounded BoC analyst.

    Unlike :func:`build_boc_news_config`, this variant does not search the web.
    Its evidence is injected by :class:`BoCDecisionPromptBuilder` from the
    cutoff-filtered document store, making historical runs reproducible.
    """
    return AgentConfig(
        name="boc_analyst_research_grounded",
        model=model,
        instruction=_BOC_ANALYST_INSTRUCTION
        + "\n6. Treat `research_evidence` as the only documentary evidence supplied for this run. "
        "Cite its document ids in `key_signals`, distinguish Bank statements from your inference, "
        "and state explicitly when the list is empty.\n",
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_boc_agent_predictor(
    config: AgentConfig,
    *,
    prompt_builder: BoCDecisionPromptBuilder | None = None,
) -> AgentPredictor:
    """Wrap an :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`BoCDecisionPromptBuilder` and the
    :class:`~aieng.forecasting.methods.agentic.outputs.CategoricalAgentForecastOutput`
    schema, which converts the agent's cut/hold/hike distribution into a
    single
    :class:`~aieng.forecasting.evaluation.prediction.CategoricalForecast`
    prediction and preserves ``reasoning`` / ``key_signals`` in metadata for
    the reasoning-alignment evaluation.

    Parameters
    ----------
    config : AgentConfig
        Any config produced by :func:`build_boc_basic_config` or
        :func:`build_boc_news_config`.
    prompt_builder : BoCDecisionPromptBuilder or None
        Optional configured builder. The default preserves the quantitative-
        only payload; the research-grounded factory supplies a document-aware
        builder.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=prompt_builder or BoCDecisionPromptBuilder(),
        output_schema=CategoricalAgentForecastOutput,
    )


def build_boc_research_predictor(
    model: str = LITE_MODEL,
    *,
    max_documents: int = 3,
    max_chars_per_document: int = 6_000,
) -> AgentPredictor:
    """Return the cached-communications research-grounded predictor."""
    builder = BoCDecisionPromptBuilder(
        document_sources=DEFAULT_RESEARCH_SOURCES,
        max_documents=max_documents,
        max_chars_per_document=max_chars_per_document,
    )
    return build_boc_agent_predictor(build_boc_research_config(model), prompt_builder=builder)


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_boc_basic_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
