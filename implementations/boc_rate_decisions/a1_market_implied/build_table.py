"""Build the committed ``origin_date -> (p_cut, p_hold, p_hike)`` market table.

Reads the raw files fetched by :mod:`fetch_market_data`, applies
:func:`implied.implied_quote` at every origin in a spec, and writes
``data/market_implied_boc.csv`` — the table :class:`MarketImpliedPredictor`
reads at predict time. Diagnostics (both estimators, contracts, settlements,
open interest) are written alongside the probabilities so a suspicious cell can
be traced back to the exchange data it came from.

Usage
-----
``python -m boc_rate_decisions.a1_market_implied.build_table``
``python -m boc_rate_decisions.a1_market_implied.build_table --spec specs/boc_rate_direction_eval.yaml``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ..data import load_meeting_schedule
from .implied import implied_quote


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "specs" / "boc_rate_direction_eval.yaml"
OUTPUT_PATH = DATA_DIR / "market_implied_boc.csv"


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Load the fetched COA settlements and the daily CORRA series."""
    coa = pd.read_csv(DATA_DIR / "coa_settlements.csv", parse_dates=["date", "expiry_date"])
    corra_frame = pd.read_csv(DATA_DIR / "corra_daily.csv", parse_dates=["date"])
    return coa, corra_frame.set_index("date")["value"].sort_index()


def origins_from_spec(spec_path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ``(origin, meeting_date)`` pairs declared by a spec.

    The meeting date is ``origin + horizon`` days, cross-checked against the
    committed announcement calendar so a drifting spec fails loudly here rather
    than silently pricing the wrong month.
    """
    spec = yaml.safe_load(spec_path.read_text())
    horizon = int(spec["task"]["horizons"][0])
    schedule = {pd.Timestamp(d).normalize() for d in load_meeting_schedule()}

    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for raw_origin in spec["origin_dates"]:
        origin = pd.Timestamp(raw_origin).normalize()
        meeting = origin + pd.Timedelta(days=horizon)
        if meeting not in schedule:
            raise ValueError(f"Origin {origin:%Y-%m-%d} + {horizon}d = {meeting:%Y-%m-%d}, not a scheduled meeting.")
        pairs.append((origin, meeting))
    return pairs


def build(spec_path: Path) -> pd.DataFrame:
    """Compute one implied-probability row per origin in ``spec_path``."""
    coa, corra = load_raw()
    schedule = load_meeting_schedule()

    rows: list[dict[str, object]] = []
    for origin, meeting in origins_from_spec(spec_path):
        quote = implied_quote(origin, meeting, coa=coa, corra=corra, meeting_dates=schedule)
        rows.append(
            {
                "origin_date": origin.strftime("%Y-%m-%d"),
                "meeting_date": meeting.strftime("%Y-%m-%d"),
                "p_cut": round(quote.p_cut, 6),
                "p_hold": round(quote.p_hold, 6),
                "p_hike": round(quote.p_hike, 6),
                "method": quote.method,
                "corra_now": round(quote.corra_now, 4),
                "expected_move_pp": round(quote.expected_move, 4),
                "source": "TMX Montreal Exchange COA (One-Month CORRA Futures) settlement + BoC Valet AVG.INTWO",
                "diagnostics": json.dumps(quote.diagnostics, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the table for a spec and write it to ``data/market_implied_boc.csv``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Spec whose origins to price.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Where to write the table.")
    args = parser.parse_args()

    table = build(args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    display = table.drop(columns=["diagnostics", "source"])
    print(display.to_string(index=False))
    print(f"\nwrote {len(table)} rows -> {args.output}")


if __name__ == "__main__":
    main()
