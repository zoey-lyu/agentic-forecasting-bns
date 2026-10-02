"""Measure how much a router could win, and what a fixed blend already takes.

How much could a router win, and does a fixed blend already take it?

Step 0 of the turn-detection plan. A3 found that the market and the macro
logistic have complementary errors: the logistic is the better forecaster on
meetings where the Bank holds, the market on meetings where it moves. That
invites an agent that decides which to trust — but before writing one, two
numbers decide whether the idea is worth anything:

``oracle``
    Take the better of the two *per meeting*, with hindsight. No router can do
    better, so this is the ceiling on the whole direction.
``blend``
    Average the two forecasts with a weight fixed at 0.5. No agent, no
    decision, no cost. An agent has to beat this to be worth building.

If the blend already captures most of the oracle's advantage, the router
direction is dead and Step 0 has saved every step after it. The thresholds are
fixed in ``PREREGISTRATION.md``.

Averaging probabilities is not the same as averaging scores: RPS is convex, so
a pool of two forecasts is usually better than the mean of their scores and can
beat both. That is exactly why the blend has to be computed rather than
inferred from the leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..a1_market_implied.leaderboard import CATEGORY_ORDER


def linear_pool(first: pd.DataFrame, second: pd.DataFrame, weight: float = 0.5) -> pd.DataFrame:
    """Blend two probability tables: ``weight · first + (1 - weight) · second``.

    Parameters
    ----------
    first, second : pd.DataFrame
        Probability tables indexed by meeting date with one column per label in
        :data:`CATEGORY_ORDER`.
    weight : float
        Weight on ``first``. Fixed at 0.5 for the pre-registered comparison;
        other values are descriptive only.

    Returns
    -------
    pd.DataFrame
        Pooled probabilities over the meetings both tables cover.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1]; got {weight}.")
    index = first.index.intersection(second.index)
    pooled = weight * first.loc[index, CATEGORY_ORDER] + (1.0 - weight) * second.loc[index, CATEGORY_ORDER]
    return pooled.sort_index()


def oracle_scores(first: pd.Series, second: pd.Series) -> pd.Series:
    """Per-meeting best of two score series — the hindsight ceiling for a router.

    This is not achievable: picking the winner needs the outcome. It bounds
    what any rule choosing between the two forecasts could reach.
    """
    index = first.index.intersection(second.index)
    return pd.Series(np.minimum(first.loc[index].to_numpy(), second.loc[index].to_numpy()), index=index).sort_index()


def ceiling_summary(
    market: pd.Series,
    model: pd.Series,
    blend: pd.Series,
    *,
    window: str,
) -> dict[str, object]:
    """Summarise headroom and what the fixed blend captures of it.

    Parameters
    ----------
    market, model, blend : pd.Series
        Per-meeting RPS for the market reference, the competing model, and the
        0.5 pool of the two.
    window : str
        Label for the window being summarised.

    Returns
    -------
    dict
        ``headroom`` is ``RPS_market - RPS_oracle``; ``captured_share`` is the
        fraction of it the blend takes. The share is ``None`` when the headroom
        is not positive, because a ratio against a non-positive denominator
        says nothing.
    """
    index = market.index.intersection(model.index).intersection(blend.index)
    market_mean = float(market.loc[index].mean())
    oracle_mean = float(oracle_scores(market, model).loc[index].mean())
    blend_mean = float(blend.loc[index].mean())
    headroom = market_mean - oracle_mean
    captured = (market_mean - blend_mean) / headroom if headroom > 0 else None

    return {
        "window": window,
        "n_meetings": int(len(index)),
        "market": round(market_mean, 4),
        "model": round(float(model.loc[index].mean()), 4),
        "blend_0.5": round(blend_mean, 4),
        "oracle": round(oracle_mean, 4),
        "headroom": round(headroom, 4),
        "captured_share": round(captured, 4) if captured is not None else None,
    }


def weight_curve(
    market_probabilities: pd.DataFrame,
    model_probabilities: pd.DataFrame,
    outcomes: pd.Series,
    *,
    steps: int = 11,
) -> pd.DataFrame:
    """Mean RPS across blend weights — **descriptive only**.

    Reading a best weight off this curve and then reporting its score would be
    fitting on the evaluation window. The pre-registered comparison uses 0.5.
    """
    from ..a1_market_implied.leaderboard import per_meeting_rps  # noqa: PLC0415 - avoids a cycle at import time

    rows = []
    for weight in np.linspace(0.0, 1.0, steps):
        pooled = linear_pool(market_probabilities, model_probabilities, float(weight))
        rows.append(
            {
                "weight_on_market": round(float(weight), 2),
                "mean_rps": round(float(per_meeting_rps(pooled, outcomes).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["ceiling_summary", "linear_pool", "oracle_scores", "weight_curve"]


def router_requirement(subsets: pd.DataFrame, window: str = "pooled") -> tuple[pd.DataFrame, dict[str, float]]:
    """Translate the ceiling into a specification for how good a router must be.

    Headroom says how much is on the table; this says what an agent has to do
    to collect it. A router sends each meeting to the market or to the logistic,
    so its quality is two numbers — the share of moves it correctly calls live
    (sensitivity) and the share of holds it correctly calls quiet
    (specificity) — and the resulting mean RPS is a weighted mix of the four
    subset means.

    The asymmetry matters more than the level: on this data, sending a move to
    the logistic costs several times what sending a hold to the market does. A
    router should therefore be biased toward calling meetings live.

    Parameters
    ----------
    subsets : pd.DataFrame
        Output of the Step 1 subset table: ``window``, ``subset``,
        ``predictor_id``, ``n_meetings``, ``mean_rps``.
    window : str
        Which window's subset means to use.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        A break-even grid (rows: specificity, columns: sensitivity) of mean
        RPS, and the headline numbers: the market's own score, the perfect
        router's score, and the cost ratio of a miss to a false alarm.

    Notes
    -----
    Descriptive design guidance, not a pre-registered decision rule.
    """
    means = subsets[subsets["window"] == window].set_index(["subset", "predictor_id"])["mean_rps"]
    counts = subsets[subsets["window"] == window].set_index(["subset", "predictor_id"])["n_meetings"]
    n_moves = float(counts[("moves", "market_implied")])
    n_holds = float(counts[("holds", "market_implied")])
    move_market = float(means[("moves", "market_implied")])
    move_model = float(means[("moves", "boc_logistic_macro")])
    hold_market = float(means[("holds", "market_implied")])
    hold_model = float(means[("holds", "boc_logistic_macro")])
    total = n_moves + n_holds

    rows = []
    for specificity in (1.0, 0.9, 0.8, 0.7):
        row: dict[str, object] = {"specificity": specificity}
        for sensitivity in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            routed = (
                n_holds * (specificity * hold_model + (1.0 - specificity) * hold_market)
                + n_moves * (sensitivity * move_market + (1.0 - sensitivity) * move_model)
            ) / total
            row[f"sensitivity_{sensitivity}"] = round(routed, 4)
        rows.append(row)

    miss_cost = move_model - move_market
    false_alarm_cost = hold_market - hold_model
    headline = {
        "market_alone": round((n_moves * move_market + n_holds * hold_market) / total, 4),
        "perfect_router": round((n_moves * move_market + n_holds * hold_model) / total, 4),
        "miss_cost": round(miss_cost, 4),
        "false_alarm_cost": round(false_alarm_cost, 4),
        "miss_to_false_alarm_ratio": round(miss_cost / false_alarm_cost, 2) if false_alarm_cost > 0 else float("nan"),
    }
    return pd.DataFrame(rows).set_index("specificity"), headline
