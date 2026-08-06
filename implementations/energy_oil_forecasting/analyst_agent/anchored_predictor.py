"""Predictor that reconstructs forecasts from an anchor plus bounded agent signals.

:class:`AnchoredAgentPredictor` is an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
variant whose ``predict()`` does not trust the agent's raw output directly.
Instead the agent emits two bounded signals per horizon
(:class:`~energy_oil_forecasting.analyst_agent.agent.AnchoredForecastOutput`)
relative to a precomputed statistical anchor
(:class:`~energy_oil_forecasting.analyst_agent.anchor_lookup.AnchorSource`) it
is shown in the prompt, and the final point forecast / quantile grid is
reconstructed here in the harness:

    final_point       = anchor_point + w_loc * signal_loc * anchor_half_width
    final_half_width  = anchor_half_width * (1 + w_width * signal_width)

Both ``w_loc`` and ``w_width`` are fixed constructor weights. ``w_loc`` defaults
to 0.2, fit from the 51-origin 2025 backtest (see
``implementations/energy_oil_forecasting/anchor_weight_fitting.ipynb`` — pooled
mean CRPS across two independent baseline-agent runs is minimized near
``w_loc=0`` and rises monotonically as ``w_loc`` grows, so 0.2 sits inside the
notebook's recommended 0.1-0.25 range rather than at the Day-1 placeholder of
0.5). ``w_width`` stays at its Day-1 placeholder of 0.5 — that notebook found
the free-form baseline agent's own quantile spread was narrower than the
anchor's in 287/290 pooled predictions, leaving no variance in the calm 2025
window to fit against; see ``planning-docs/anchor-modifier-safety-day1-plan.md``
for the full writeup. No matter what the agent reports, the final point can
never move further than ``w_loc * anchor_half_width`` from the anchor, and the
final width can never shrink below the anchor's own width (``signal_width`` has
no negative half).

This subclasses rather than edits the shared ``AgentPredictor`` /
``ContinuousAgentForecastOutput`` — those are used by other predictors that
should not get this behavior. See the day-1 plan's "Integration point"
section for the rationale.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.langfuse_traces import stamp_forecast_on_trace
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY, AgentConfig
from aieng.forecasting.methods.agentic.predictor import _run_coroutine_sync
from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence, trace_url_for
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.analyst_agent.agent import (
    AnchoredForecastOutput,
    AnchoredWtiPromptBuilder,
    build_wti_anchored_config,
)
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorSource
from energy_oil_forecasting.analyst_agent.news_cache import NewsCacheSource
from pydantic import ValidationError


logger: logging.Logger = logging.getLogger(__name__)


class AnchoredAgentPredictor(AgentPredictor):
    """``AgentPredictor`` that reconstructs forecasts from anchor + bounded signals.

    ``predict()`` is overridden rather than inherited: the base class calls
    ``output.to_predictions()``, which has no way to receive the anchor
    lookup table (the ``ForecastPromptBuilder``/``AgentForecastOutput``
    protocols don't carry it), and
    :meth:`~energy_oil_forecasting.analyst_agent.agent.AnchoredForecastOutput.to_predictions`
    raises ``NotImplementedError`` for exactly this reason.

    Parameters
    ----------
    agent_config : AgentConfig
        Typically built via :func:`~energy_oil_forecasting.analyst_agent.agent.build_wti_anchored_config`.
    prompt_builder : AnchoredWtiPromptBuilder
        Must share the same ``anchor_source`` passed here, so the agent sees
        exactly the anchors this predictor reconstructs against.
    anchor_source : AnchorSource
        Precomputed anchor lookup table.
    w_loc : float, default=0.2
        Bounds how far the final point forecast may move from the anchor, as
        a fraction of the anchor's half-width. Fit from the 2025 backtest —
        see the module docstring.
    w_width : float, default=0.5
        Bounds how much the final interval may widen beyond the anchor's own
        width. Set to ``0.0`` to force the final width to equal the anchor's
        exactly (the "Location-only" row of the Day-2 ablation matrix). Still
        unfit — see the module docstring.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        prompt_builder: AnchoredWtiPromptBuilder,
        *,
        anchor_source: AnchorSource,
        w_loc: float = 0.2,
        w_width: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Wire the base ``AgentPredictor`` to ``AnchoredForecastOutput`` and store reconstruction state."""
        super().__init__(agent_config, prompt_builder, output_schema=AnchoredForecastOutput, **kwargs)
        self.anchor_source = anchor_source
        self.w_loc = w_loc
        self.w_width = w_width

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Run the agent, then reconstruct predictions from anchor + bounded signals.

        Parameters
        ----------
        task : ForecastingTask
            Defines the prediction problem — target series, horizon(s),
            frequency, and resolution logic.
        context : ForecastContext
            The information state available at forecast time.

        Returns
        -------
        list[Prediction]
            One ``Prediction`` per horizon in ``task.horizons``. An empty
            list is returned when the agent's structured output cannot be
            converted (the error is logged); schema validation errors on the
            agent's raw JSON are not swallowed.
        """
        prompt = self.prompt_builder(task=task, context=context)
        initial_state = {AS_OF_STATE_KEY: str(context.as_of)[:10]}
        output_str = _run_coroutine_sync(self._runner.run_text_async(prompt, initial_state=initial_state))
        output_str = strip_markdown_fence(output_str)

        try:
            output: AnchoredForecastOutput = self.output_schema.model_validate_json(output_str)
        except ValidationError:
            try:
                output = self.output_schema.model_validate(json.loads(output_str))
            except Exception:
                logger.warning("Raw agent response (schema validation failed):\n%s", output_str)
                raise

        try:
            predictions = self._reconstruct_predictions(output, task=task, context=context)
        except Exception as e:
            logger.error("Error reconstructing predictions from anchor + signals: %s", e)
            return []

        trace_id = self._runner.last_trace_id
        if trace_id is not None:
            trace_url = trace_url_for(trace_id)
            for prediction in predictions:
                prediction.metadata.setdefault("langfuse_trace_id", trace_id)
                if trace_url is not None:
                    prediction.metadata.setdefault("langfuse_trace_url", trace_url)
            stamp_forecast_on_trace(predictions, trace_id=trace_id)

        return predictions

    def _reconstruct_predictions(
        self,
        output: AnchoredForecastOutput,
        *,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Turn validated bounded signals into ``Prediction`` objects via the anchor.

        Raises
        ------
        ValueError
            If the signal horizons don't exactly match ``task.horizons``.
        """
        by_horizon = {signal.horizon: signal for signal in output.signals}
        expected = set(task.horizons)
        actual = set(by_horizon)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"Anchored agent output must contain exactly the task horizons. Missing: {missing}; extra: {extra}"
            )

        issued_at = datetime.utcnow()  # naive UTC; Prediction.issued_at expects timezone-naive
        offset = pd.tseries.frequencies.to_offset(task.frequency)

        predictions: list[Prediction] = []
        for horizon in task.horizons:
            signal = by_horizon[horizon]
            anchor = self.anchor_source.get(as_of=context.as_of, horizon=horizon)

            final_point = anchor.point_forecast + self.w_loc * signal.signal_loc * anchor.half_width
            final_half_width = anchor.half_width * (1 + self.w_width * signal.signal_width)
            scale = final_half_width / anchor.half_width
            final_quantiles = {
                q: final_point + (v - anchor.point_forecast) * scale for q, v in anchor.quantiles.items()
            }

            metadata: dict[str, Any] = {
                "anchor_point": anchor.point_forecast,
                "anchor_half_width": anchor.half_width,
                "signal_loc": signal.signal_loc,
                "signal_width": signal.signal_width,
                "w_loc": self.w_loc,
                "w_width": self.w_width,
                "final_point": final_point,
                "final_half_width": final_half_width,
            }
            if output.rationale.strip():
                metadata["rationale"] = output.rationale
            if signal.rationale.strip():
                metadata["horizon_rationale"] = signal.rationale
            if output.key_signals:
                metadata["key_signals"] = list(output.key_signals)

            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=final_point, quantiles=final_quantiles),
                    metadata=metadata,
                )
            )

        return predictions


def build_wti_anchored_predictor(
    anchor_source: AnchorSource,
    *,
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    w_loc: float = 0.2,
    w_width: float = 0.5,
    news_source: NewsCacheSource | None = None,
    anchor_prompt: str = "original",
    **config_kwargs: Any,
) -> AnchoredAgentPredictor:
    """Wire :func:`~energy_oil_forecasting.analyst_agent.agent.build_wti_anchored_config` into an :class:`AnchoredAgentPredictor`.

    Mirrors :func:`~energy_oil_forecasting.analyst_agent.agent.build_wti_agent_predictor`.

    Parameters
    ----------
    anchor_source : AnchorSource
        Precomputed anchor lookup table, e.g. from
        ``AnchorSource.from_spec_id("energy_oil_eval")``. Shared between the
        prompt builder (so the agent sees the anchor) and the predictor (so
        reconstruction uses the same anchor).
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Ignored when
        ``news_source`` is supplied.
    w_loc : float, default=0.2
        See :class:`AnchoredAgentPredictor`.
    w_width : float, default=0.5
        See :class:`AnchoredAgentPredictor`.
    news_source : NewsCacheSource or None
        When supplied, passed to both the prompt builder (so the agent sees
        the cached briefing) and the config factory (so it drops the live
        search tool to match) — see
        :func:`~energy_oil_forecasting.analyst_agent.agent.build_wti_anchored_config`.
    anchor_prompt : str, default="original"
        Which anchor-prompt framing the agent is given — ``"original"``,
        ``"symloc"``, or ``"twosided"``. Framing-only; the reconstruction
        arithmetic here is identical for all three, and each config is renamed
        so the variants' predictions cache separately. See
        :func:`~energy_oil_forecasting.analyst_agent.agent.build_wti_anchored_config`.
    **config_kwargs : Any
        Forwarded to :func:`~energy_oil_forecasting.analyst_agent.agent.build_wti_anchored_config`
        (e.g. ``verifier_model``).

    Returns
    -------
    AnchoredAgentPredictor
    """
    config = build_wti_anchored_config(
        model=model,
        search_model=search_model,
        news_source=news_source,
        anchor_prompt=anchor_prompt,
        **config_kwargs,
    )
    return AnchoredAgentPredictor(
        config,
        AnchoredWtiPromptBuilder(anchor_source=anchor_source, news_source=news_source),
        anchor_source=anchor_source,
        w_loc=w_loc,
        w_width=w_width,
    )
