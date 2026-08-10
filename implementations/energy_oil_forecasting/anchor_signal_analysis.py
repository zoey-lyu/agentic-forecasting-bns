"""Cross-regime analysis of the anchored agent's bounded signals.

Everything here replays *stored* predictions. The anchored predictor persists
``signal_loc``, ``signal_width``, ``anchor_point`` and ``anchor_half_width`` in
each prediction's metadata, and the anchor quantile grid lives in the anchor
table, so the whole reconstruction

    final_point = anchor_point + w_loc * signal_loc * anchor_half_width
    scale       = 1 + w_width * signal_width
    quantiles   = final_point + (anchor_q - anchor_point) * scale

can be recomputed at any weights without re-running an agent. That makes weight
sweeps, channel ablations and controls free once a run exists.

Two families of measurement live here, and the distinction matters:

**Label-free** (:func:`gate_table`) compares an anchored arm to the *same model's*
free-form run on the same cells. It never reads a realised price, so iterating
prompts against it cannot overfit the evaluation window.

**Label-reading** (:func:`weight_sweep`, :func:`channel_ablation`,
:func:`channel_control`, :func:`bootstrap_ci`) scores against realised outcomes
and should be looked at sparingly and pre-registered.

The controls exist because of a measured failure. On the 2026 window the width
channel beat "no adjustment" with every bootstrap CI excluding zero -- and then
scored identically when every cell's ``signal_width`` was replaced by its own
mean. The significant channel was a constant recalibration of the anchor's
intervals, not agent judgement. Any claim that a signal earns its place has to
clear :func:`channel_control`, not just :func:`bootstrap_ci`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import energy_oil_forecasting
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorSource
from energy_oil_forecasting.data import build_wti_service


ROOT = Path(energy_oil_forecasting.__file__).parent
STEM = "_continuous__wti_oil_price_forecast.yaml"

#: Short label -> model id. ``preview`` is the *weaker* of the two.
MODELS: dict[str, str] = {
    "preview": "gemini-3.1-flash-lite-preview",
    "3.5-flash": "gemini-3.5-flash",
}

#: Arm label -> predictor-id template. ``free-form`` sees no anchor and emits a
#: price; its signal is derived, which is what makes it the reference for
#: :func:`gate_table`.
ARMS: dict[str, str] = {
    "free-form": "agent_predictor_wti_analyst_news_cached_{m}",
    "original": "agent_predictor_wti_analyst_anchored_cached_news_{m}",
    "symloc": "agent_predictor_wti_analyst_anchored_cached_news_symloc_{m}",
    "twosided": "agent_predictor_wti_analyst_anchored_cached_news_twosided_{m}",
}

#: Weight grid used by :func:`weight_sweep`.
DEFAULT_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def crps(quantiles: dict[float, float], actual: float) -> float:
    """Pinball-loss CRPS approximation over a quantile grid.

    Parameters
    ----------
    quantiles : dict[float, float]
        Mapping from quantile level to forecast value.
    actual : float
        Realised value.

    Returns
    -------
    float
    """
    return 2 * float(np.mean([max(q * (actual - v), (q - 1) * (actual - v)) for q, v in quantiles.items()]))


def _actuals(as_of: datetime | None = None) -> dict[pd.Timestamp, float]:
    """Realised WTI closes keyed by normalised timestamp."""
    service = build_wti_service()
    resolved = as_of or datetime.now(tz=timezone.utc).replace(tzinfo=None)
    raw = service.get_series("wti_crude_oil_price", as_of=resolved)
    return {pd.Timestamp(r["timestamp"]).normalize(): float(r["value"]) for _, r in raw.iterrows()}


def load_arm(spec_id: str, arm: str, model: str, actuals: dict | None = None) -> pd.DataFrame | None:
    """Load one arm's per-cell signals, anchor, and realised value.

    Parameters
    ----------
    spec_id : str
        ``"energy_oil_eval"`` (2026) or ``"energy_oil_backtest"`` (2025).
    arm : str
        A key of :data:`ARMS`.
    model : str
        A key of :data:`MODELS`.
    actuals : dict or None
        Realised closes; fetched once if omitted. Pass it in when looping.

    Returns
    -------
    pandas.DataFrame or None
        Indexed by ``(as_of, horizon)``. ``None`` when that run has not been
        computed yet, so callers can render a partial grid rather than crash.

    Notes
    -----
    For the free-form arm ``signal_loc`` is *derived* --
    ``(agent_point - anchor_point) / anchor_half_width`` -- which is exactly what
    ``signal_loc`` denotes for the anchored arms, putting both output formats on
    one axis. ``signal_width`` is derived the same way from the 90% interval.
    """
    path = ROOT / f"data/predictions/{spec_id}/{ARMS[arm].format(m=MODELS[model])}{STEM}"
    if not path.exists():
        return None
    actuals = _actuals() if actuals is None else actuals
    close_series = pd.Series(actuals).sort_index()

    with open(ROOT / f"specs/{spec_id}.yaml") as f:
        spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
    task = spec.tasks[0]
    src = AnchorSource.from_spec_id(spec_id)
    offset = pd.tseries.frequencies.to_offset(task.frequency)

    with open(path) as f:
        raw = yaml.safe_load(f)
    preds = raw["predictions"] if isinstance(raw, dict) and "predictions" in raw else raw

    rows: list[dict[str, Any]] = []
    for r in preds:
        as_of, fd = pd.Timestamp(r["as_of"]), pd.Timestamp(r["forecast_date"])
        horizon = {(as_of + offset * h): h for h in task.horizons}.get(fd)
        actual = actuals.get(fd.normalize())
        if horizon is None or actual is None:
            continue
        try:
            anchor = src.get(as_of=r["as_of"], horizon=horizon)
        except KeyError:
            continue  # market-holiday gap in the anchor table; see the notebook
        meta = r.get("metadata") or {}
        q = {float(k): v for k, v in r["payload"]["quantiles"].items()}
        if "signal_loc" in meta:
            loc, width = meta["signal_loc"], meta["signal_width"]
        else:
            loc = (r["payload"]["point_forecast"] - anchor.point_forecast) / anchor.half_width
            width = ((q[0.95] - q[0.05]) / 2) / anchor.half_width - 1
        rows.append(
            {
                "as_of": as_of.date(),
                "horizon": horizon,
                "signal_loc": loc,
                "signal_width": width,
                "anchor_pt": anchor.point_forecast,
                "anchor_hw": anchor.half_width,
                "anchor_q": anchor.quantiles,
                "agent_pt": r["payload"]["point_forecast"],
                "agent_q": q,
                "origin_close": _origin_close(as_of, anchor.point_forecast, close_series),
                "actual": actual,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).set_index(["as_of", "horizon"]).sort_index()


def _origin_close(as_of: pd.Timestamp, fallback: float, series: pd.Series) -> float:
    """The close on the origin day itself.

    Deliberately *not* what the forecast context contains. ``get_series(as_of=X)``
    correctly stops at the previous close -- standing at the origin you do not yet
    know that day's close -- and the anchor is fit on exactly that. But the cached
    news briefing has cutoff ``as_of`` and routinely states the current level, so a
    news-reading agent knows this number and the anchor does not.

    That is a legitimate edge rather than lookahead (a real forecaster at the origin
    does see overnight trading), but it means the agent's apparent deviation from the
    anchor contains a component that is not judgement. :func:`decompose_baseline`
    exists to separate the two.

    ``series`` is the close series, passed in so it is built once per load rather
    than per row.

    Takes the last close **at or before** the origin, never the next one. Four of the
    eval origins fall on market holidays (Presidents' Day, Memorial Day), where there
    is no origin-day close at all: reaching forward to the next trading day would hand
    this baseline a price the news could not have reported. Reaching backward instead
    returns the previous close -- which is exactly what the anchor was fit on -- so a
    holiday origin correctly contributes nothing to the measured lag.
    """
    window = series[series.index <= as_of]
    return float(window.iloc[-1]) if len(window) else fallback


def decompose_baseline(spec_id: str, model: str, actuals: dict | None = None) -> pd.DataFrame:
    """Split the news baseline's advantage over the anchor into three contributions.

    Four forecasts differing by exactly one ingredient, scored on the same cells:

    ==  =========================  =========================  ==========================
    id  centre                     interval width             adds
    ==  =========================  =========================  ==========================
    A   AutoARIMA's point          AutoARIMA's interval       nothing -- no agent at all
    B   the origin day's close     AutoARIMA's interval       the price level the news states
    C   the origin day's close     the **agent's** interval   the agent's uncertainty sizing
    D   the **agent's** point      the **agent's** interval   its actual news judgement
    ==  =========================  =========================  ==========================

    ``D - C`` is the question the whole project rests on: what does the news
    reasoning add *beyond* restating a price the briefing already gave away?

    Parameters
    ----------
    spec_id, model : str
        See :func:`load_arm`.
    actuals : dict or None
        Passed through to :func:`load_arm`.

    Returns
    -------
    pandas.DataFrame
        One row per cell with columns ``A``, ``B``, ``C``, ``D`` (per-cell CRPS)
        and ``as_of``, ready for :func:`contribution_ci`.

    Notes
    -----
    On the 51-origin 2025 window the only significant positive contribution is
    ``C - B`` -- the agent's narrower intervals -- while ``D - C`` is *positive*
    (worse) on both models and significantly so on the weaker one.
    """
    d = load_arm(spec_id, "free-form", model, actuals)
    if d is None:
        return pd.DataFrame()
    rows = []
    for (as_of, _), r in zip(d.index, d.to_dict("records"), strict=True):
        oc, pt, an = r["origin_close"], r["agent_pt"], r["anchor_pt"]
        rows.append(
            {
                "as_of": as_of,
                "A": crps(r["anchor_q"], r["actual"]),
                "B": crps({k: oc + (v - an) for k, v in r["anchor_q"].items()}, r["actual"]),
                "C": crps({k: oc + (v - pt) for k, v in r["agent_q"].items()}, r["actual"]),
                "D": crps(r["agent_q"], r["actual"]),
            }
        )
    return pd.DataFrame(rows)


def contribution_ci(df: pd.DataFrame, col_a: str, col_b: str, *, n_boot: int = 10_000,
                    seed: int = 42) -> tuple[float, float, float]:
    """Origin-clustered bootstrap CI for ``mean(col_a - col_b)`` on a decomposition frame.

    Returns
    -------
    tuple of float
        ``(mean_difference, ci_low, ci_high)``. Negative means ``col_a`` scores better.
    """
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"as_of": df["as_of"], "x": df[col_a] - df[col_b]})
    by_origin = {o: g["x"].to_numpy() for o, g in d.groupby("as_of")}
    origins = sorted(by_origin)
    draws = np.array(
        [np.concatenate([by_origin[o] for o in rng.choice(origins, len(origins))]).mean()
         for _ in range(n_boot)]
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(d["x"].mean()), float(lo), float(hi)


def sign_only_control(df: pd.DataFrame, *, w_loc: float = 0.2, w_width: float = 0.5) -> pd.Series:
    """Does the agent's location *magnitude* earn anything beyond its *direction*?

    Replaces ``signal_loc`` with its sign, rescaling ``w_loc`` by ``mean|signal_loc|``
    so the average dollar move is unchanged -- without that, sign-only would move
    several times further and the test would measure size rather than information.

    Returns
    -------
    pandas.Series
        ``real``, ``sign_only``, and ``sign_minus_real``.
    """
    loc = df["signal_loc"].to_numpy()
    magnitude = float(np.abs(loc).mean())
    real = float(score(df, w_loc, w_width).mean())
    sign_only = float(score(df, w_loc * magnitude, w_width, loc=np.sign(loc)).mean())
    return pd.Series({"real": real, "sign_only": sign_only, "sign_minus_real": sign_only - real})


def trailing_vol(as_of: Any, actuals: dict | None = None) -> float:
    """Annualised realised volatility (%) over the 21 trading days ending at ``as_of``."""
    series = pd.Series(actuals or _actuals()).sort_index()
    window = series[series.index <= pd.Timestamp(as_of)].iloc[-22:]
    return float(np.std(np.diff(np.log(window.to_numpy()))) * np.sqrt(252) * 100)


def naive_signal(df: pd.DataFrame) -> np.ndarray:
    """Signal a zero-judgement agent would emit by merely restating the origin day's close.

    ``(origin day's close - anchor point) / anchor half-width``. Comparing an arm's
    mean ``signal_loc`` against this separates "the model leans up because of the
    news" from "the anchor is one day stale and the window rose" -- the latter
    accounts for essentially all of ``gemini-3.5-flash``'s measured lean on 2026.
    """
    return (df["origin_close"].to_numpy() - df["anchor_pt"].to_numpy()) / df["anchor_hw"].to_numpy()


def directional_edge(df: pd.DataFrame, *, reference: str = "anchor_pt") -> pd.Series:
    """Directional hit rate against a null matched to the arm's own lean.

    Parameters
    ----------
    df : pandas.DataFrame
        From :func:`load_arm`.
    reference : {"anchor_pt", "origin_close"}
        What "up" is measured against. ``anchor_pt`` asks "will it land above the
        ARIMA forecast?"; ``origin_close`` asks "will the price rise from here?",
        which is the question a news agent that never sees the anchor was actually
        asked. Scoring an unanchored arm against ``anchor_pt`` measures a question
        it was never posed.

    Returns
    -------
    pandas.Series
        ``n``, ``says_up``, ``hit_rate``, ``null``, ``edge``.

    Notes
    -----
    The null is *not* 0.5. An arm that says "up" 97% of the time in a window that
    rose 60% of the time scores ~0.60 while knowing nothing, so the baseline is
    ``P(says up) * P(truth up) + P(says down) * P(truth down)``.
    """
    ref = df[reference].to_numpy()
    sig = np.sign(df["agent_pt"].to_numpy() - ref) if reference == "origin_close" \
        else np.sign(df["signal_loc"].to_numpy())
    truth = np.sign(df["actual"].to_numpy() - ref)
    keep = sig != 0
    sig, truth = sig[keep], truth[keep]
    says_up, truth_up = float((sig > 0).mean()), float((truth > 0).mean())
    hit = float((sig == truth).mean())
    null = says_up * truth_up + (1 - says_up) * (1 - truth_up)
    return pd.Series({"n": len(sig), "says_up": says_up, "hit_rate": hit, "null": null,
                      "edge": hit - null})


def score(df: pd.DataFrame, w_loc: float, w_width: float, *, loc: np.ndarray | None = None,
          width: np.ndarray | None = None) -> np.ndarray:
    """Per-cell CRPS of the reconstruction at the given weights.

    Parameters
    ----------
    df : pandas.DataFrame
        From :func:`load_arm`.
    w_loc, w_width : float
        Harness weights to replay at.
    loc, width : numpy.ndarray or None
        Optional signal overrides, used by :func:`channel_control` to substitute
        a constant or a permutation for the stored signal.

    Returns
    -------
    numpy.ndarray
        One CRPS per row of ``df``.
    """
    loc = df["signal_loc"].to_numpy() if loc is None else loc
    width = df["signal_width"].to_numpy() if width is None else width
    out = []
    for (_, r), sl, sw in zip(df.iterrows(), loc, width, strict=True):
        point = r["anchor_pt"] + w_loc * sl * r["anchor_hw"]
        scale = 1 + w_width * sw
        out.append(crps({lvl: point + (v - r["anchor_pt"]) * scale for lvl, v in r["anchor_q"].items()},
                        r["actual"]))
    return np.array(out)


def gate_table(spec_id: str, model: str, actuals: dict | None = None) -> pd.DataFrame:
    """Label-free comparison of each anchored arm against the model's own view.

    The free-form arm is the same model on the same news with no anchor shown,
    so it is that model's judgement transmitted losslessly -- the ceiling an
    anchored prompt is trying to reach.

    Parameters
    ----------
    spec_id, model : str
        See :func:`load_arm`.
    actuals : dict or None
        Passed through to :func:`load_arm`.

    Returns
    -------
    pandas.DataFrame
        One row per available arm. ``sd_ratio`` targets 1.0 (range restored)
        and ``mean_gap`` targets 0.0 (no manufactured lean). Realised prices are
        never read.
    """
    actuals = _actuals() if actuals is None else actuals
    arms = {a: load_arm(spec_id, a, model, actuals) for a in ARMS}
    arms = {a: d for a, d in arms.items() if d is not None}
    if "free-form" not in arms:
        return pd.DataFrame()

    idx = None
    for d in arms.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    ff = arms["free-form"].loc[idx, "signal_loc"].clip(-1, 1)

    rows = []
    for arm, d in arms.items():
        v = d.loc[idx, "signal_loc"]
        v = v.clip(-1, 1) if arm == "free-form" else v
        rows.append(
            {
                "arm": arm,
                "n": len(idx),
                "mean": v.mean(),
                "sd": v.std(),
                "pct_negative": 100 * (v < 0).mean(),
                "sd_ratio": np.nan if arm == "free-form" else v.std() / ff.std(),
                "mean_gap": np.nan if arm == "free-form" else v.mean() - ff.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("arm")


def weight_sweep(df: pd.DataFrame, *, sweep: str = "w_loc", held: float = 0.5,
                 grid: tuple[float, ...] = DEFAULT_GRID) -> pd.Series:
    """Mean CRPS across a weight grid, holding the other weight fixed.

    Parameters
    ----------
    df : pandas.DataFrame
        From :func:`load_arm`.
    sweep : {"w_loc", "w_width"}
        Which weight to vary.
    held : float
        Value of the other weight.
    grid : tuple of float
        Weights to evaluate.

    Returns
    -------
    pandas.Series
        Mean CRPS indexed by weight.

    Notes
    -----
    A monotone sweep running to the grid boundary is *not* evidence the weight
    should be raised. On a window whose truth sat above the anchor most of the
    time, a rectified (mostly-positive) signal is flattered by any increase in
    ``w_loc``, so the sweep measures the regime. Compare across regimes and run
    :func:`channel_control` before acting on a sweep.
    """
    vals = {
        w: score(df, w, held).mean() if sweep == "w_loc" else score(df, held, w).mean()
        for w in grid
    }
    return pd.Series(vals, name="mean_crps").rename_axis(sweep)


def channel_ablation(df: pd.DataFrame, *, w_loc: float = 0.2, w_width: float = 0.5) -> pd.Series:
    """Mean CRPS with each signal channel switched off in turn.

    Returns
    -------
    pandas.Series
        ``anchor`` (both off), ``loc only``, ``width only``, ``full``.
    """
    return pd.Series(
        {
            "anchor": score(df, 0.0, 0.0).mean(),
            "loc only": score(df, w_loc, 0.0).mean(),
            "width only": score(df, 0.0, w_width).mean(),
            "full": score(df, w_loc, w_width).mean(),
        },
        name="mean_crps",
    )


def channel_control(df: pd.DataFrame, channel: str, *, w_loc: float = 0.2, w_width: float = 0.5,
                    n_shuffle: int = 20, seed: int = 7) -> pd.Series:
    """Does the agent's *per-cell* judgement on a channel earn anything?

    Replaces the channel's stored signal with two null versions carrying no
    cell-level information:

    - ``constant`` -- every cell gets the signal's own mean, so only the level
      survives
    - ``shuffled`` -- signals permuted across cells, so the distribution
      survives but the pairing with the news does not

    Parameters
    ----------
    df : pandas.DataFrame
        From :func:`load_arm`.
    channel : {"loc", "width"}
        Which channel to null out.
    w_loc, w_width : float
        Weights to evaluate at.
    n_shuffle : int
        Permutations averaged for the shuffled control.
    seed : int
        Seed for the permutation RNG.

    Returns
    -------
    pandas.Series
        ``real``, ``constant``, ``shuffled``, and ``real_minus_constant``.
        A ``real_minus_constant`` near zero means the channel is delivering a
        level shift (or a blanket widening), not judgement.
    """
    rng = np.random.default_rng(seed)
    loc, width = df["signal_loc"].to_numpy(), df["signal_width"].to_numpy()
    target = loc if channel == "loc" else width
    const = np.full(len(df), target.mean())

    def run(replacement: np.ndarray) -> float:
        kw = {"loc": replacement} if channel == "loc" else {"width": replacement}
        return float(score(df, w_loc, w_width, **kw).mean())

    real = float(score(df, w_loc, w_width).mean())
    constant = run(const)
    shuffled = float(np.mean([run(rng.permutation(target)) for _ in range(n_shuffle)]))
    return pd.Series(
        {"real": real, "constant": constant, "shuffled": shuffled,
         "real_minus_constant": real - constant},
        name=channel,
    )


def bootstrap_ci(df: pd.DataFrame, a: tuple[float, float], b: tuple[float, float], *,
                 n_boot: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Origin-clustered bootstrap CI for ``CRPS(a) - CRPS(b)``.

    Parameters
    ----------
    df : pandas.DataFrame
        From :func:`load_arm`.
    a, b : tuple of float
        ``(w_loc, w_width)`` for each side of the comparison.
    n_boot : int
        Bootstrap replicates.
    seed : int
        RNG seed.

    Returns
    -------
    tuple of float
        ``(mean_difference, ci_low, ci_high)``.

    Notes
    -----
    The horizons at one origin share a news briefing and an anchor fit, so they
    are not independent draws. Resampling *origins* rather than cells is what
    keeps the interval honest; resampling cells understates it badly.
    """
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"as_of": [i[0] for i in df.index], "d": score(df, *a) - score(df, *b)})
    by_origin = {o: g["d"].to_numpy() for o, g in d.groupby("as_of")}
    origins = sorted(by_origin)
    draws = np.array(
        [np.concatenate([by_origin[o] for o in rng.choice(origins, len(origins))]).mean()
         for _ in range(n_boot)]
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(d["d"].mean()), float(lo), float(hi)
