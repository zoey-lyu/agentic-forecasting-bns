"""Run the missing cells of the cached-news grid (prompt variant x model) on a backtest spec.

Defaults to ``energy_oil_eval`` (2026, 18 origins); pass ``--spec energy_oil_backtest``
for the calm 2025 window (51 origins). Running both is the only way to tell a real
weight from a regime artifact -- the ``w_loc`` sweep points in opposite directions
on the two windows, so anything fitted on one alone is fitting the regime.


The 2026 free-form baselines already on disk were run against *live* ``search_web``,
while the anchored run used the ``NewsCacheSource``. That confound makes
"anchored says 'up' 97% of the time vs. the free-form agent's 74%" impossible to
attribute to the bounded-signal schema rather than to the two agents simply
having read different articles.

This script fills in the three missing cells so all four share byte-identical news:

              | preview (lite)          | 3.5-flash
    anchored  | already on disk         | run here
    free-form | run here                | run here

Usage
-----
    uv run python scripts/run_cached_news_2x2.py --smoke      # 2 origins, all 3 cells
    uv run python scripts/run_cached_news_2x2.py             # full 18-origin eval spec
    uv run python scripts/run_cached_news_2x2.py --only free_form_preview
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import energy_oil_forecasting
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest
from energy_oil_forecasting.analyst_agent.agent import (
    build_wti_agent_predictor,
    build_wti_news_config,
)
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorSource
from energy_oil_forecasting.analyst_agent.anchored_predictor import build_wti_anchored_predictor
from energy_oil_forecasting.analyst_agent.news_cache import NewsCacheSource
from energy_oil_forecasting.data import build_wti_service


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_cached_news_2x2")

LITE = "gemini-3.1-flash-lite-preview"
FLASH = "gemini-3.5-flash"


def build_cells(anchor_source: AnchorSource, news: NewsCacheSource) -> dict:
    """Return the three predictor factories keyed by cell name."""
    return {
        # The original prompt on both models. The preview cell predates this
        # script (its 2026 predictions were produced by hand) and is declared
        # here so the grid can be reproduced end-to-end on any spec.
        "anchored_preview": lambda: build_wti_anchored_predictor(
            anchor_source, model=LITE, news_source=news
        ),
        "anchored_flash": lambda: build_wti_anchored_predictor(
            anchor_source, model=FLASH, news_source=news
        ),
        # Same anchors, same news, same reconstruction -- only the prompt's
        # framing of signal_loc's negative half differs, so a change in the
        # paired up/down pass-through ratios is attributable to wording.
        "anchored_preview_symloc": lambda: build_wti_anchored_predictor(
            anchor_source, model=LITE, news_source=news, anchor_prompt="symloc"
        ),
        "anchored_flash_symloc": lambda: build_wti_anchored_predictor(
            anchor_source, model=FLASH, news_source=news, anchor_prompt="symloc"
        ),
        # Third framing: forces a written bearish case before a number is
        # picked, and drops the one-directional "geopolitical risk" vocabulary.
        # One prompt for both models on purpose -- a per-model prompt would
        # confound prompt with model. See _ANCHOR_SUPPLEMENT_TWO_SIDED.
        "anchored_preview_twosided": lambda: build_wti_anchored_predictor(
            anchor_source, model=LITE, news_source=news, anchor_prompt="twosided"
        ),
        "anchored_flash_twosided": lambda: build_wti_anchored_predictor(
            anchor_source, model=FLASH, news_source=news, anchor_prompt="twosided"
        ),
        "free_form_preview": lambda: build_wti_agent_predictor(
            build_wti_news_config(model=LITE, news_source=news), news_source=news
        ),
        "free_form_flash": lambda: build_wti_agent_predictor(
            build_wti_news_config(model=FLASH, news_source=news), news_source=news
        ),
    }


def main() -> None:
    """Run the requested 2x2 cells and report per-cell prediction counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="use energy_oil_eval_smoke (2 origins)")
    parser.add_argument(
        "--spec",
        default="energy_oil_eval",
        help="backtest spec to run: energy_oil_eval (2026, 18 origins) or energy_oil_backtest (2025, 51). "
        "Overridden by --smoke.",
    )
    parser.add_argument("--only", help="run a single cell by name")
    parser.add_argument("--force-refresh", action="store_true", help="recompute even if cached")
    args = parser.parse_args()

    spec_id = "energy_oil_eval_smoke" if args.smoke else args.spec
    root = Path(energy_oil_forecasting.__file__).parent
    with open(root / f"specs/{spec_id}.yaml") as f:
        spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

    data_service = build_wti_service()
    # The smoke spec's origins are a subset of the eval spec's, so its anchors
    # live in the eval table -- there is no separate smoke anchor file.
    anchor_spec_id = "energy_oil_eval" if spec_id == "energy_oil_eval_smoke" else spec_id
    anchor_source = AnchorSource.from_spec_id(anchor_spec_id)
    news = NewsCacheSource()

    cells = build_cells(anchor_source, news)
    if args.only:
        if args.only not in cells:
            parser.error(f"--only must be one of {sorted(cells)}")
        cells = {args.only: cells[args.only]}

    for name, factory in cells.items():
        predictor = factory()
        logger.info("=== %s -> predictor_id=%s (spec=%s) ===", name, predictor.predictor_id, spec_id)
        results = cached_multi_backtest(predictor, spec, data_service, force_refresh=args.force_refresh)
        for task_id, result in results.items():
            logger.info(
                "%s [%s]: n=%d mean_crps=%.4f", name, task_id, len(result.predictions), result.mean_score
            )
        if not results:
            logger.warning("%s produced NO results", name)


if __name__ == "__main__":
    main()
