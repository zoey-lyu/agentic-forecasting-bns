"""Record every leakage-verifier verdict so its decisions can be read by hand.

Searching in 2026 for a 2025 rate decision will surface articles that state the
outcome, so the verifier standing between the search provider and the agent is
what the whole search line rests on. Across 25 search-enabled runs it never
rejected a result: no retry warning, no ``[SEARCH_VERIFICATION_FAILED]``. That
is either a verifier passing genuinely clean text or a verifier that is not
biting, and from the outside the two look identical — the verdicts are logged
at INFO and the removed spans are not persisted at all.

This script wraps ``_verify_no_leakage`` and writes every verdict to disk:
the query, the cutoff, the text under review, and each removal with its quote
and reason. Production code is untouched; the wrapper lives here.

    uv run python scripts/run_verifier_audit.py [--origins ...] [--out DIR]

Read the output. The question it answers is not "did it fire" but "when it
passed something, should it have".
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)

import boc_rate_decisions  # noqa: E402
from aieng.forecasting.evaluation import BacktestSpec  # noqa: E402
from aieng.forecasting.evaluation.backtest import run_eval_loop  # noqa: E402
from aieng.forecasting.methods.agentic import agent_factory  # noqa: E402
from aieng.forecasting.models import LITE_MODEL  # noqa: E402
from boc_rate_decisions.analyst_agent import build_boc_agent_predictor, build_boc_news_config  # noqa: E402
from boc_rate_decisions.data import build_boc_service  # noqa: E402


PKG = Path(boc_rate_decisions.__file__).parent
DEFAULT_ORIGINS = ["2025-01-01", "2025-08-20"]

VERDICTS: list[dict] = []


def _install_recorder() -> None:
    """Wrap the module-level verifier so each verdict is captured verbatim."""
    original = agent_factory._verify_no_leakage

    async def recording_verifier(**kwargs):  # type: ignore[no-untyped-def]
        verdict = await original(**kwargs)
        VERDICTS.append(
            {
                "query": kwargs.get("query"),
                "cutoff_date": kwargs.get("cutoff_date"),
                "text": kwargs.get("text"),
                "clean": verdict.clean,
                "confidence": verdict.confidence,
                "removals": [
                    {"quote": r.quote, "reason": r.reason, "basis": r.basis} for r in verdict.removals
                ],
            }
        )
        return verdict

    agent_factory._verify_no_leakage = recording_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", nargs="*", default=DEFAULT_ORIGINS)
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--out", default=str(PKG / "data" / "verifier_audit"))
    args = parser.parse_args()

    # The per-attempt verdict line is logged at INFO; the default level hides it.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("aieng.forecasting.methods.agentic.agent_factory").setLevel(logging.INFO)

    _install_recorder()

    spec = BacktestSpec.model_validate(
        yaml.safe_load((PKG / "specs" / "boc_rate_direction_eval.yaml").read_text())
    )
    service = build_boc_service()
    origins = [datetime.fromisoformat(o) for o in args.origins]

    predictor = build_boc_agent_predictor(build_boc_news_config(model=args.model))
    preds, _, skipped = run_eval_loop(
        predictor=predictor,
        task=spec.task,
        origins=origins,
        warmup=spec.warmup,
        data_service=service,
        max_retries=1,
        retry_delay=2.0,
    )
    print(f"\nforecasts produced: {len(preds)}   skipped: {skipped}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "verdicts.jsonl").open("w") as f:
        for v in VERDICTS:
            f.write(json.dumps(v) + "\n")

    total_removals = sum(len(v["removals"]) for v in VERDICTS)
    print("\n=== verifier activity ===")
    print(f"  verifier calls           {len(VERDICTS)}")
    print(f"  verdicts marked clean    {sum(1 for v in VERDICTS if v['clean'])}")
    print(f"  total spans removed      {total_removals}")
    if VERDICTS:
        print(f"  mean confidence          {sum(v['confidence'] for v in VERDICTS) / len(VERDICTS):.1f} / 10")
        print(f"  mean chars reviewed      {sum(len(v['text'] or '') for v in VERDICTS) / len(VERDICTS):.0f}")

    for v in VERDICTS:
        if v["removals"]:
            print(f"\n  cutoff {v['cutoff_date']}  query: {str(v['query'])[:90]}")
            for r in v["removals"]:
                print(f"    - [{r['basis']}] {r['quote'][:160]}")
                print(f"      reason: {r['reason'][:160]}")

    print(f"\nwrote {out_dir / 'verdicts.jsonl'}  (read the `text` fields by hand)")


if __name__ == "__main__":
    main()
