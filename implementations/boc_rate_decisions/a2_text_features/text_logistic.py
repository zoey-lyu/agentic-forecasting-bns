"""Logistic baseline extended with LLM-extracted statement-stance features.

Same estimator, same fit-at-origin protocol and the same four macro features as
:class:`~boc_rate_decisions.predictors.logistic_baseline.BoCLogisticPredictor`;
the only change is three extra columns describing the most recent statement
(see :mod:`features`). That is the point: with everything else held fixed, the
difference between the two runs *is* the value of the text.

Three arms are produced by one class, so nothing but the feature block differs
between them:

``use_text=False``
    The ablation. Must reproduce the existing macro baseline exactly — the run
    script checks this, because a silent divergence would make the comparison
    meaningless.
``use_text=True``
    The A2 arm.
``use_text=True, shuffle_seed=N``
    The control. Same three features and the same parameter count, but each
    meeting gets another meeting's reading. Any gain that survives this was
    never about what the statements said.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory

from ..data import (
    BOND_YIELD_2YR_SERIES_ID,
    CPI_SERIES_ID,
    DIRECTION_TASK_CATEGORIES,
    TARGET_RATE_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
)
from ..predictors.logistic_baseline import FEATURE_NAMES, build_feature_row
from .features import TEXT_FEATURE_NAMES, StanceTable


class BoCTextLogisticPredictor(Predictor):
    """Fit-at-origin multinomial logistic on macro plus statement-stance features.

    Parameters
    ----------
    stance : StanceTable or None
        Source of the text features. Required when ``use_text`` is True.
    use_text : bool
        Include the stance features. ``False`` gives the macro-only ablation.
    regularization_c : float
        Inverse regularization strength for scikit-learn's ``LogisticRegression``.
        Left at the baseline's untuned 1.0 — tuning it here would confound the
        text comparison with a hyperparameter change.
    min_training_examples : int
        Minimum resolved past meetings with computable features required to fit;
        below this the predictor falls back to class frequencies, as the macro
        baseline does.
    predictor_suffix : str or None
        Appended to ``predictor_id`` so several arms can share one leaderboard.
    """

    def __init__(
        self,
        stance: StanceTable | None = None,
        *,
        use_text: bool = True,
        regularization_c: float = 1.0,
        min_training_examples: int = 16,
        predictor_suffix: str | None = None,
    ) -> None:
        if use_text and stance is None:
            raise ValueError("use_text=True requires a StanceTable.")
        self._stance = stance
        self._use_text = use_text
        self._c = regularization_c
        self._min_train = min_training_examples
        self._suffix = predictor_suffix

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature columns this arm fits on."""
        return [*FEATURE_NAMES, *TEXT_FEATURE_NAMES] if self._use_text else list(FEATURE_NAMES)

    @property
    def predictor_id(self) -> str:
        """Stable identifier, distinguishing the arms."""
        base = "boc_logistic_macro_text" if self._use_text else "boc_logistic_macro"
        return f"{base}_{self._suffix}" if self._suffix else base

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit on past meetings visible at the origin and emit one forecast.

        Raises
        ------
        ValueError
            If the task is not a single-horizon categorical task.
        """
        if task.payload_type != "categorical":
            raise ValueError(f"{type(self).__name__} requires a categorical task; got '{task.payload_type}'.")
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} supports exactly one horizon; got {task.horizons}.")

        as_of = pd.Timestamp(context.as_of)
        series = {
            "target": context.get_series(task.target_series_id),
            "rate": context.get_series(TARGET_RATE_SERIES_ID),
            "yield": context.get_series(BOND_YIELD_2YR_SERIES_ID),
            "cpi": context.get_series(CPI_SERIES_ID),
            "unemployment": context.get_series(UNEMPLOYMENT_SERIES_ID),
        }
        lead = pd.tseries.frequencies.to_offset(task.frequency) * task.horizons[0]

        feature_rows, outcomes = self._training_data(series, lead)
        current = self._feature_row(as_of, series)
        categories = task.categories if task.categories is not None else DIRECTION_TASK_CATEGORIES

        payload, info = self._fit_and_predict(categories, feature_rows, outcomes, current)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=(as_of + lead).to_pydatetime(),
                payload=payload,
                metadata={"n_train": len(outcomes), "features_used": self.feature_names, **info},
            )
        ]

    def _feature_row(self, origin: pd.Timestamp, series: dict[str, pd.DataFrame]) -> dict[str, float] | None:
        """Macro features at ``origin``, plus stance features when enabled."""
        macro = build_feature_row(origin, series["rate"], series["yield"], series["cpi"], series["unemployment"])
        if macro is None:
            return None
        if not self._use_text:
            return macro
        assert self._stance is not None  # noqa: S101 - guaranteed by __init__
        text = self._stance.feature_row(origin)
        if text is None:
            return None
        return {**macro, **text}

    def _training_data(
        self, series: dict[str, pd.DataFrame], lead: pd.DateOffset
    ) -> tuple[list[list[float]], list[float]]:
        """Rebuild every resolved past meeting's features at its own origin."""
        feature_rows: list[list[float]] = []
        outcomes: list[float] = []
        for meeting, outcome in zip(series["target"]["timestamp"], series["target"]["value"], strict=True):
            row = self._feature_row(pd.Timestamp(meeting) - lead, series)
            if row is None:
                continue
            feature_rows.append([row[name] for name in self.feature_names])
            outcomes.append(float(outcome))
        return feature_rows, outcomes

    def _fit_and_predict(
        self,
        categories: list[TaskCategory],
        feature_rows: list[list[float]],
        outcomes: list[float],
        current: dict[str, float] | None,
    ) -> tuple[CategoricalForecast, dict[str, object]]:
        """Fit the multinomial model, falling back to class frequencies."""
        degenerate = (
            current is None or len(outcomes) < self._min_train or len(set(outcomes)) < 2  # noqa: PLR2004
        )
        if degenerate:
            return CategoricalForecast(probabilities=self._class_frequencies(outcomes, categories)), {
                "model": "class_frequency_fallback"
            }

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        model = make_pipeline(StandardScaler(), LogisticRegression(C=self._c, max_iter=1000))
        model.fit(np.asarray(feature_rows), np.asarray(outcomes))

        assert current is not None  # noqa: S101 - degenerate check above
        x_now = np.asarray([[current[name] for name in self.feature_names]])
        row = model.predict_proba(x_now)[0]

        probabilities = {category.label: 0.0 for category in categories}
        for class_value, probability in zip(model.classes_, row, strict=True):
            probabilities[self._category_for(float(class_value), categories).label] = float(probability)

        return CategoricalForecast(probabilities=probabilities), {
            "model": "multinomial_logistic_regression",
            "features": dict(zip(self.feature_names, (float(v) for v in x_now[0]), strict=True)),
        }

    def _class_frequencies(self, outcomes: list[float], categories: list[TaskCategory]) -> dict[str, float]:
        """Empirical category frequencies over the visible outcomes."""
        if not outcomes:
            uniform = 1.0 / len(categories)
            return {category.label: uniform for category in categories}
        counts = {category.label: 0 for category in categories}
        for outcome in outcomes:
            counts[self._category_for(outcome, categories).label] += 1
        return {label: count / len(outcomes) for label, count in counts.items()}

    def _category_for(self, value: float, categories: list[TaskCategory]) -> TaskCategory:
        """Find the task category matching an observed class value."""
        for category in categories:
            if np.isclose(value, category.value):
                return category
        raise ValueError(f"Observed class value {value} is not declared in task.categories.")


__all__ = ["BoCTextLogisticPredictor"]
