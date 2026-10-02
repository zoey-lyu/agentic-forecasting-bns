"""Test whether text adds over a market-priced baseline across 2009-2026.

Every test in this folder so far has run on 28 meetings, because the CORRA
futures that make the market reference only start in 2023. But the question
underneath — does reading the Bank's statements tell you something the priced
information does not — does not require futures. It requires a market-priced
baseline, and the 2-year Government of Canada yield has been in this repo's
data since 1991.

So the baseline here is the existing four-feature model, whose strongest
feature is the 2-year yield spread: the bond market's own read on where policy
is going. The test adds the frozen lexicon on top. Both are fit at every origin
on meetings that had resolved by then, the same protocol as everywhere else,
and both are scored on whether the Bank moved.

This is a weaker claim than beating the futures curve and is labelled that way
wherever it appears. It is also the only version of the question with roughly
130 meetings behind it instead of 28.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..predictors.logistic_baseline import build_feature_row


#: Lexicon columns added by the text arm.
TEXT_FEATURES = ["lex2_tilt", "lex2_intensity", "lex2_hedging"]

#: Meetings that must have resolved before a fit is attempted.
MIN_TRAINING_MEETINGS = 24


def build_panel(
    meetings: pd.DataFrame,
    lexicon: pd.DataFrame,
    series: dict[str, pd.DataFrame],
    *,
    lead_days: int = 28,
) -> pd.DataFrame:
    """Assemble one row per meeting: outcome, macro features, lexicon features.

    Features are rebuilt at each meeting's own origin (``meeting - lead_days``)
    so training and prediction see the same information shape, and the lexicon
    row is the most recent statement published by that origin.

    Parameters
    ----------
    meetings : pd.DataFrame
        Canonical direction series (``timestamp``, ``value``).
    lexicon : pd.DataFrame
        Frozen v2 lexicon scores indexed by statement date.
    series : dict[str, pd.DataFrame]
        The macro series keyed ``rate``, ``yield``, ``cpi``, ``unemployment``.
    lead_days : int
        Forecast lead.

    Returns
    -------
    pd.DataFrame
        Indexed by meeting date; ``moved`` plus every feature column. Meetings
        without a full feature set are dropped.
    """
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for timestamp, value in zip(meetings["timestamp"], meetings["value"], strict=True):
        meeting = pd.Timestamp(timestamp)
        origin = meeting - pd.Timedelta(days=lead_days)
        macro = build_feature_row(origin, series["rate"], series["yield"], series["cpi"], series["unemployment"])
        if macro is None:
            continue
        visible = lexicon[lexicon.index <= origin]
        if visible.empty:
            continue
        text = visible.iloc[-1]
        rows[meeting] = {
            "moved": float(float(value) != 0.0),
            **macro,
            **{column: float(text[column]) for column in TEXT_FEATURES},
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def fit_at_origin_probabilities(panel: pd.DataFrame, features: list[str]) -> pd.Series:
    """Predict each meeting from a model fitted only on the meetings before it.

    An expanding window, refit at every meeting. Meetings with too little
    history behind them, or whose training set has a single outcome class, fall
    back to the running base rate — the same fallback the macro baseline uses.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    predictions: dict[pd.Timestamp, float] = {}
    for position, meeting in enumerate(panel.index):
        train = panel.iloc[:position]
        if len(train) < MIN_TRAINING_MEETINGS or train["moved"].nunique() < 2:  # noqa: PLR2004
            predictions[meeting] = float(train["moved"].mean()) if len(train) else 0.5
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(train[features].to_numpy(), train["moved"].to_numpy())
        predictions[meeting] = float(model.predict_proba(panel.loc[[meeting], features].to_numpy())[0, 1])
    return pd.Series(predictions).sort_index()


def shuffled_panel(panel: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a copy with the lexicon columns permuted across meetings."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(panel))
    shuffled = panel.copy()
    for column in TEXT_FEATURES:
        shuffled[column] = panel[column].to_numpy()[order]
    return shuffled


def era_of(meeting: pd.Timestamp) -> str:
    """Bucket a meeting into one of four policy eras, fixed in advance."""
    if meeting < pd.Timestamp("2015-01-01"):
        return "2009-2014"
    if meeting < pd.Timestamp("2020-01-01"):
        return "2015-2019"
    if meeting < pd.Timestamp("2023-01-01"):
        return "2020-2022"
    return "2023-2026"


__all__ = [
    "MIN_TRAINING_MEETINGS",
    "TEXT_FEATURES",
    "build_panel",
    "era_of",
    "fit_at_origin_probabilities",
    "shuffled_panel",
]
