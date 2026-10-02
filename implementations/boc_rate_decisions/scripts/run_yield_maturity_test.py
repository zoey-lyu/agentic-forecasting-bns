"""Which maturity is the better read on market expectations: 2-year or 1-year?

``build_feature_row`` uses the 2-year Government of Canada yield minus the
policy rate as its market-expectations feature. A screen over 132 origins found
the 1-year Treasury bill spread separated cut-origins from hike-origins more
cleanly (2.67 pooled SDs against the 2-year's 1.95). A screen is not a result,
so this runs the comparison on the actual task: same logistic baseline, same
120 origins, three-way RPS, one series swapped.

The swap is done by writing the 1-year series into the store under the
2-year's id, so every downstream consumer — feature builder, predictor, cutoff
enforcement — behaves identically and the only difference is the numbers. The
1-year series comes from the Bank's own Valet API (free, no key), is weekly
rather than daily, and is stamped ``released_at = timestamp + 1 day`` to match
the convention the 2-year carries.

Nothing here touches production code or the shared prediction store, and no
LLM is called: the logistic baseline is deterministic.

    uv run python scripts/run_yield_maturity_test.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)

import boc_rate_decisions  # noqa: E402
from aieng.forecasting.evaluation import BacktestSpec, backtest  # noqa: E402
from boc_rate_decisions.data import BOND_YIELD_2YR_SERIES_ID, build_boc_service  # noqa: E402
from boc_rate_decisions.gate import bootstrap_ci, rps  # noqa: E402
from boc_rate_decisions.predictors import BoCLogisticPredictor  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent
VALET_URL = "https://www.bankofcanada.ca/valet/observations/group/tbill_all/json?start_date=2008-01-01"
RELEASE_LAG_DAYS = 1


def fetch_tbill(label: str) -> pd.DataFrame:
    """One Treasury-bill maturity from Valet, as a canonical series frame.

    Valet publishes the Tuesday and Wednesday readings as separate series that
    share a label; both are kept and averaged per date, which in practice means
    taking whichever one exists on a given day.
    """
    with urllib.request.urlopen(VALET_URL, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    ids = [k for k, v in payload["seriesDetail"].items() if v["label"] == label]
    if not ids:
        raise KeyError(f"no Valet series labelled {label!r}")
    rows = []
    for obs in payload["observations"]:
        values = [float(obs[i]["v"]) for i in ids if i in obs and obs[i].get("v")]
        if values:
            rows.append({"timestamp": pd.Timestamp(obs["d"]), "value": float(np.mean(values))})
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    frame["released_at"] = frame["timestamp"] + pd.Timedelta(days=RELEASE_LAG_DAYS)
    return frame


def score(spec: BacktestSpec, service, labels: list[str], values: list[float]) -> tuple[np.ndarray, list]:
    """Backtest the logistic baseline and return per-origin RPS plus origins."""
    result = backtest(predictor=BoCLogisticPredictor(), spec=spec, data_service=service)
    truth = service.get_series(spec.task.target_series_id, pd.Timestamp.now().to_pydatetime())
    by_date = {pd.Timestamp(t).normalize(): v for t, v in zip(truth["timestamp"], truth["value"])}
    idx = {v: i for i, v in enumerate(values)}
    probs, outcomes, origins = [], [], []
    for p in result.predictions:
        probs.append([float(p.payload.probabilities[c]) for c in labels])
        outcomes.append(idx[int(by_date[pd.Timestamp(p.forecast_date).normalize()])])
        origins.append(pd.Timestamp(p.as_of).date())
    return rps(np.array(probs), np.array(outcomes)), origins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="boc_rate_direction_backtest.yaml")
    args = parser.parse_args()

    spec = BacktestSpec.model_validate(yaml.safe_load((PKG / "specs" / args.spec).read_text()))
    labels = [c.label for c in spec.task.categories]
    values = [c.value for c in spec.task.categories]

    print("baseline: 2-year GoC yield (what build_feature_row uses today)")
    baseline_service = build_boc_service()
    baseline_cells, origins = score(spec, baseline_service, labels, values)
    print(f"  n={len(baseline_cells)}  mean RPS={baseline_cells.mean():.4f}\n")

    for maturity in ("1 year", "6 month", "3 month"):
        service = build_boc_service()
        frame = fetch_tbill(maturity)
        service._store.put(BOND_YIELD_2YR_SERIES_ID, frame, service._store.get_metadata(BOND_YIELD_2YR_SERIES_ID))
        cells, swapped_origins = score(spec, service, labels, values)
        if swapped_origins != origins:
            print(f"  {maturity}: origin sets differ ({len(swapped_origins)} vs {len(origins)}); skipping comparison")
            continue
        delta, lo, hi = bootstrap_ci(cells, baseline_cells)
        verdict = "better" if hi < 0 else ("worse" if lo > 0 else "no difference")
        print(
            f"swap in {maturity:8s} T-bill:  RPS={cells.mean():.4f}   "
            f"delta vs 2-year {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]   {verdict}"
        )


if __name__ == "__main__":
    main()
