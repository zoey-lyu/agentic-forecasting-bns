"""Run the stance extraction over every cached BoC statement, once.

Writes ``data/statement_stance.csv`` — one row per statement, committed so the
downstream predictor is reproducible without re-spending LLM calls. The run is
resumable: statements already in the CSV are skipped unless ``--force``.

Every record is checked before it is written: the ``evidence`` quote must
appear verbatim in the statement it came from. A failed check does not abort
the run (one bad row out of 141 should not cost the other 140), but the row is
flagged in ``evidence_verbatim`` and the count is reported at the end, because
a stance score whose supporting quote is invented is not a measurement.

Usage
-----
``python -m boc_rate_decisions.a2_text_features.extract_stance_table``
``python -m boc_rate_decisions.a2_text_features.extract_stance_table --limit 5``
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from aieng.forecasting.models import LITE_MODEL

from .stance import GUIDANCE_SCORES, extract_stance, quote_is_verbatim


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "reports" / "boc_press_releases"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "statement_stance.csv"

_COLUMNS = [
    "statement_date",
    "hawk_dove",
    "guidance",
    "guidance_score",
    "drivers",
    "evidence",
    "evidence_verbatim",
    "model",
    "extracted_at",
]


def statement_paths(cache_dir: Path) -> dict[pd.Timestamp, Path]:
    """Map statement date -> cached markdown path, sorted ascending."""
    return {pd.Timestamp(path.stem.removesuffix("_en")): path for path in sorted(cache_dir.glob("*_en.md"))}


def main() -> None:
    """Extract stances for all cached statements not already recorded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=LITE_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N new extractions (smoke runs).")
    parser.add_argument("--force", action="store_true", help="Re-extract statements already in the table.")
    args = parser.parse_args()

    paths = statement_paths(args.cache_dir)
    if not paths:
        raise SystemExit(f"No cached statements under {args.cache_dir}. Run scripts/fetch_boc_press_releases.py.")

    existing = pd.DataFrame(columns=_COLUMNS)
    if args.output.exists() and not args.force:
        existing = pd.read_csv(args.output)
        done = set(pd.to_datetime(existing["statement_date"]))
        paths = {date: path for date, path in paths.items() if date not in done}

    print(f"{len(paths)} statements to extract (model={args.model})", flush=True)
    rows: list[dict[str, object]] = []
    total_cost = 0.0
    for index, (date, path) in enumerate(paths.items(), start=1):
        if args.limit is not None and index > args.limit:
            break
        text = path.read_text()
        try:
            stance, cost = extract_stance(text, model=args.model)
        except Exception as error:  # noqa: BLE001 - one bad statement must not lose the rest
            print(f"  [{date:%Y-%m-%d}] FAILED: {error}", file=sys.stderr, flush=True)
            continue
        total_cost += cost
        verbatim = quote_is_verbatim(stance.evidence, text)
        rows.append(
            {
                "statement_date": f"{date:%Y-%m-%d}",
                "hawk_dove": stance.hawk_dove,
                "guidance": stance.guidance,
                "guidance_score": GUIDANCE_SCORES[stance.guidance],
                "drivers": ";".join(stance.drivers),
                "evidence": stance.evidence.replace("\n", " ").strip(),
                "evidence_verbatim": verbatim,
                "model": args.model,
                "extracted_at": datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            }
        )
        flag = "" if verbatim else "  <- quote not verbatim"
        print(f"  [{date:%Y-%m-%d}] {stance.hawk_dove:+d} {stance.guidance}{flag}", flush=True)

    table = pd.concat([existing, pd.DataFrame(rows, columns=_COLUMNS)], ignore_index=True)
    table = table.sort_values("statement_date").drop_duplicates(subset=["statement_date"], keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    bad = int((~table["evidence_verbatim"].astype(bool)).sum())
    print(f"\nwrote {len(table)} rows -> {args.output}")
    print(f"new extractions: {len(rows)}   reported cost: ${total_cost:.4f}   non-verbatim quotes: {bad}")


if __name__ == "__main__":
    main()
