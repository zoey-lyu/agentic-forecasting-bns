"""Score whatever the live forecast series has accumulated so far.

Reads ``live_forecasts.jsonl``, joins the announcements that have since
resolved, and reports Brier per arm. Records for meetings that have not
happened yet are listed as pending.

The pre-registration forbids using these numbers to adjust anything until the
series reaches twenty records — reading them is fine, acting on them is what
would destroy the only uncontaminated evidence in the project. The script
prints that reminder with the count so it is hard to forget.

Usage
-----
``python -m boc_rate_decisions.turn_detection.score_live``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..data import DIRECTION_SERIES_ID, build_boc_service
from .live_forecast import LIVE_RECORDS
from .move_task import brier_scores
from .run_steps import USE_CASE_DIR


#: Below this many resolved records, the series states no verdict. See PREREGISTRATION.md.
CLAIM_THRESHOLD = 20


def main() -> None:
    """Report the live series' status and, where resolved, its scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None, help="Cutoff for reading outcomes (default: today).")
    parser.add_argument("--statcan-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "statcan")
    parser.add_argument("--fred-cache-dir", type=Path, default=USE_CASE_DIR / "data" / "fred")
    args = parser.parse_args()

    if not LIVE_RECORDS.exists():
        print(f"No live records yet at {LIVE_RECORDS}.")
        print("The first is due at origin 2026-09-30, for the 2026-10-28 announcement.")
        return

    records = [json.loads(line) for line in LIVE_RECORDS.read_text().splitlines() if line.strip()]
    # A later record for the same origin supersedes an earlier one.
    latest = {record["origin"]: record for record in records}

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    service = build_boc_service(statcan_cache_dir=args.statcan_cache_dir, fred_cache_dir=args.fred_cache_dir)
    directions = service.get_series(DIRECTION_SERIES_ID, as_of=as_of)
    outcomes = pd.Series(
        (directions["value"].astype(float) != 0.0).astype(float).to_numpy(),
        index=pd.to_datetime(directions["timestamp"]).to_numpy(),
    )

    rows = []
    pending = []
    for record in latest.values():
        meeting = pd.Timestamp(record["meeting"])
        if meeting not in outcomes.index:
            pending.append(record["meeting"])
            continue
        rows.append(
            {
                "meeting": meeting,
                "moved": outcomes.loc[meeting],
                "market": record["market"]["p_move"],
                "agent": record["agent"]["p_move"],
                "agent_no_statement": record["agent_no_statement"]["p_move"],
            }
        )

    print(f"live records: {len(latest)}   resolved: {len(rows)}   pending: {sorted(pending)}")
    if not rows:
        print("\nNothing to score yet.")
        return

    frame = pd.DataFrame(rows).set_index("meeting").sort_index()
    print("\n" + frame.to_string())
    print("\nBrier so far (lower is better):")
    for arm in ("market", "agent", "agent_no_statement"):
        print(f"  {arm:<20} {float(brier_scores(frame[arm], frame['moved']).mean()):.4f}")

    if len(rows) < CLAIM_THRESHOLD:
        print(
            f"\n{len(rows)} of {CLAIM_THRESHOLD} records needed before this series states anything. "
            "Do not tune against these numbers."
        )


if __name__ == "__main__":
    main()
