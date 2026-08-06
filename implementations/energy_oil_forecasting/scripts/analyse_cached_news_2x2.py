"""Direction / conviction analysis over the cached-news 2x2 (schema x model).

Answers the question ``run_cached_news_2x2.py`` sets up: is the anchored agent's
near-total refusal to forecast *downward* (36 of 37 signals positive) a property
of the bounded-signal schema, or was it an artefact of the anchored run reading
cached news while the free-form baselines read live search results?

Free-form agents emit a point forecast, not a ``signal_loc``. To put both schemas
on one axis, the free-form point is converted to the *implied* signal it would
have needed::

    implied = (agent_point - anchor_point) / anchor_half_width

which is exactly the quantity ``signal_loc`` denotes for the anchored schema.
``gap`` is the same normalisation applied to the truth. Sign agreement is then
comparable across schemas, and is reported against the "always guess the majority
direction" rule -- a constant strategy that any real directional skill must beat.

Usage
-----
    uv run python scripts/analyse_cached_news_2x2.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import energy_oil_forecasting
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorSource
from energy_oil_forecasting.data import build_wti_service
from scipy import stats


ROOT = Path(energy_oil_forecasting.__file__).parent

# label -> (spec_id, prediction filename). Grouped so the 2x2 reads off the table.
RUNS: dict[str, tuple[str, str]] = {
    # --- cached news, 2026 eval: the clean 2x2 ---
    "2026 anchored  / preview  [cached]": (
        "energy_oil_eval",
        "agent_predictor_wti_analyst_anchored_cached_news_gemini-3.1-flash-lite-preview_continuous__wti_oil_price_forecast.yaml",
    ),
    "2026 anchored  / 3.5-flash [cached]": (
        "energy_oil_eval",
        "agent_predictor_wti_analyst_anchored_cached_news_gemini-3.5-flash_continuous__wti_oil_price_forecast.yaml",
    ),
    "2026 free-form / preview  [cached]": (
        "energy_oil_eval",
        "agent_predictor_wti_analyst_news_cached_gemini-3.1-flash-lite-preview_continuous__wti_oil_price_forecast.yaml",
    ),
    "2026 free-form / 3.5-flash [cached]": (
        "energy_oil_eval",
        "agent_predictor_wti_analyst_news_cached_gemini-3.5-flash_continuous__wti_oil_price_forecast.yaml",
    ),
    # --- the original live-search runs, kept as the confounded comparison ---
    "2026 free-form / preview  [live]": (
        "energy_oil_eval",
        "agent_predictor_wti_analyst_news_gemini-3.1-flash-lite-preview_continuous__wti_oil_price_forecast.yaml",
    ),
    "2026 free-form / 3.5-flash [live]": (
        "energy_oil_eval",
        "agent_predictor_wti_analyst_news_gemini-3.5-flash_continuous__wti_oil_price_forecast.yaml",
    ),
    # --- the calm 2025 window, which is where the conviction pattern died ---
    "2025 free-form / preview  [live]": (
        "energy_oil_backtest",
        "agent_predictor_wti_analyst_news_gemini-3.1-flash-lite-preview_continuous__wti_oil_price_forecast.yaml",
    ),
    "2025 free-form / lite     [live]": (
        "energy_oil_backtest",
        "agent_predictor_wti_analyst_news_gemini-3.1-flash-lite_continuous__wti_oil_price_forecast.yaml",
    ),
}

BUCKETS = [(1e-9, 0.15), (0.15, 0.35), (0.35, 1e9)]
BUCKET_LABELS = ["low <0.15", "mid .15-.35", "high >0.35"]


def load_rows(spec_id: str, filename: str, actuals: pd.DataFrame) -> list[dict]:
    """Return one row per scored (origin, horizon), with implied signal and realised gap."""
    path = ROOT / f"data/predictions/{spec_id}/{filename}"
    if not path.exists():
        return []
    with open(ROOT / f"specs/{spec_id}.yaml") as f:
        spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
    task = spec.tasks[0]
    src = AnchorSource.from_spec_id(spec_id)
    offset = pd.tseries.frequencies.to_offset(task.frequency)

    with open(path) as f:
        raw = yaml.safe_load(f)
    preds = raw["predictions"] if isinstance(raw, dict) and "predictions" in raw else raw

    rows: list[dict] = []
    for r in preds:
        as_of, fd = pd.Timestamp(r["as_of"]), pd.Timestamp(r["forecast_date"])
        horizon = {(as_of + offset * h): h for h in task.horizons}.get(fd)
        if horizon is None:
            continue
        try:
            anchor = src.get(as_of=r["as_of"], horizon=horizon)
        except Exception:
            continue  # anchor-table horizon gap -- known open bug, see day-1 plan
        match = actuals[actuals["timestamp"] == fd]
        if match.empty:
            continue
        actual = float(match["value"].iloc[0])
        meta = r.get("metadata") or {}
        implied = (
            meta["signal_loc"]
            if "signal_loc" in meta
            else (r["payload"]["point_forecast"] - anchor.point_forecast) / anchor.half_width
        )
        rows.append(
            {
                "as_of": as_of.date(),
                "horizon": horizon,
                "implied": implied,
                "gap": (actual - anchor.point_forecast) / anchor.half_width,
            }
        )
    return rows


def main() -> None:
    """Print the direction, bucket, and correlation tables for every available run."""
    argparse.ArgumentParser(description=__doc__).parse_args()

    service = build_wti_service()
    actuals = service.get_series(
        "wti_crude_oil_price", as_of=datetime.now(tz=timezone.utc).replace(tzinfo=None)
    ).copy()
    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"])

    data = {label: load_rows(spec_id, fn, actuals) for label, (spec_id, fn) in RUNS.items()}
    missing = [label for label, rows in data.items() if not rows]
    data = {label: rows for label, rows in data.items() if rows}

    hdr = f"{'run':38s}{'n':>5s}{'sign agr':>10s}{'says up':>9s}{'const rule':>12s}"
    print("\n=== directional skill vs. the constant 'always guess the majority direction' rule ===")
    print(hdr)
    print("-" * len(hdr))
    for label, rows in data.items():
        nz = [r for r in rows if abs(r["implied"]) > 1e-9]
        agr = np.mean([np.sign(r["implied"]) == np.sign(r["gap"]) for r in nz])
        up = np.mean([r["implied"] > 0 for r in nz])
        truth_up = np.mean([r["gap"] > 0 for r in rows])
        const = max(truth_up, 1 - truth_up)
        flag = "  <-- beats it" if agr > const else ""
        print(f"{label:38s}{len(nz):5d}{agr:10.2f}{up:9.2f}{const:12.2f}{flag}")

    print("\n=== sign agreement by conviction bucket ===")
    print(f"{'run':38s}" + "".join(f"{b:>18s}" for b in BUCKET_LABELS))
    for label, rows in data.items():
        cells = ""
        for lo, hi in BUCKETS:
            b = [r for r in rows if lo <= abs(r["implied"]) < hi]
            cells += (
                f"{np.mean([np.sign(r['implied']) == np.sign(r['gap']) for r in b]):.2f} (n={len(b)})".rjust(18)
                if b
                else "-".rjust(18)
            )
        print(f"{label:38s}{cells}")

    print("\n=== does |implied signal| predict |realised gap|?  (the premise adaptive w_loc needs) ===")
    for label, rows in data.items():
        s = np.array([abs(r["implied"]) for r in rows])
        d = np.array([abs(r["gap"]) for r in rows])
        rho = stats.spearmanr(s, d)
        star = " *" if rho.pvalue < 0.05 else ""
        print(f"  {label:38s} rho={rho.statistic:+.3f}  p={rho.pvalue:.4f}  n={len(s)}{star}")

    if missing:
        print("\n(not on disk yet: " + ", ".join(missing) + ")")


if __name__ == "__main__":
    main()
