"""Download the raw market data A1 needs: CORRA futures, CORRA, and the target rate.

Three sources, all public and free:

1. **Montréal Exchange (TMX) historical data** — daily settlement prices for
   ``COA``, the One-Month CORRA Futures. Each contract settles on the average
   daily CORRA over its calendar contract month, so ``100 - settlement`` is the
   market's expected average overnight rate for that month. This is the
   Canadian analogue of the Fed Funds futures that CME FedWatch uses.
   The endpoint (``/en/trading/data/historical?...&dnld=1``) returns CSV and
   caps a response at ~500 rows, so requests are chunked by month.
2. **Bank of Canada Valet ``AVG.INTWO``** — daily CORRA, the rate the futures
   settle on. Needed for the "rate already in effect at the origin" leg of the
   implied-probability decomposition, and to validate expired contracts.
3. **Bank of Canada Valet ``V39079``** — the daily target for the overnight
   rate, used to date each decision's effective day and to size a 25 bp step.

Everything is written to ``a1_market_implied/data/`` as CSV and committed, so
the downstream A1 code is reproducible without network access.

Usage
-----
``python -m boc_rate_decisions.a1_market_implied.fetch_market_data``
(add ``--force`` to re-download files that already exist).
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd


MX_HISTORICAL_URL = "https://www.m-x.ca/en/trading/data/historical"
VALET_OBSERVATIONS_URL = "https://www.bankofcanada.ca/valet/observations/{series_id}/csv"

CORRA_SERIES_ID = "AVG.INTWO"
"""Valet series: Canadian Overnight Repo Rate Average (CORRA), daily, percent."""

TARGET_RATE_SERIES_ID = "V39079"
"""Valet series: Bank of Canada target for the overnight rate, daily, percent."""

CORRA_FUTURES_ROOT = "COA"
"""MX root symbol for One-Month CORRA Futures."""

DATA_DIR = Path(__file__).resolve().parent / "data"

_USER_AGENT = "Mozilla/5.0 (compatible; agentic-forecasting-bns/1.0; research)"
_REQUEST_PAUSE_SECONDS = 1.0

#: The exchange intermittently answers a chunk with 502/503. A single transient
#: failure used to abandon a 45-month fetch, so each request is retried with a
#: growing pause before the run is allowed to fail.
_MAX_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 4.0

#: Columns kept from the MX CSV. The raw file carries 28 columns of quote-level
#: detail that A1 never reads.
_MX_KEEP_COLUMNS = {
    "Date": "date",
    "Symbol": "symbol",
    "Root Symbol": "root_symbol",
    "Expiry Date": "expiry_date",
    "Settlement Price": "settlement_price",
    "Open Interest": "open_interest",
    "Volume": "volume",
}


def _http_get(url: str, *, timeout: int = 120) -> bytes:
    """GET a URL with a browser-ish User-Agent, retrying transient failures.

    Raises
    ------
    RuntimeError
        If every attempt fails; the last error is chained.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
                payload: bytes = response.read()
            return payload
        except (urllib.error.URLError, TimeoutError) as error:  # noqa: PERF203 - retry loop
            last_error = error
            if attempt == _MAX_ATTEMPTS:
                break
            pause = _RETRY_BACKOFF_SECONDS * attempt
            print(f"    request failed ({error}); retry {attempt}/{_MAX_ATTEMPTS - 1} in {pause:.0f}s", flush=True)
            time.sleep(pause)
    raise RuntimeError(f"Giving up on {url} after {_MAX_ATTEMPTS} attempts.") from last_error


def fetch_coa_settlements(start: str, end: str) -> pd.DataFrame:
    """Download daily COA settlement prices between two dates.

    Parameters
    ----------
    start, end : str
        Inclusive ``YYYY-MM-DD`` bounds.

    Returns
    -------
    pd.DataFrame
        Columns ``date``, ``symbol``, ``root_symbol``, ``expiry_date``,
        ``settlement_price``, ``open_interest``, ``volume``; one row per
        (trading day, listed contract), sorted and de-duplicated.
    """
    months = pd.date_range(start=pd.Timestamp(start).normalize(), end=pd.Timestamp(end).normalize(), freq="MS")
    # date_range on "MS" misses the partial first month when start is mid-month.
    chunk_starts = [pd.Timestamp(start)] + [m for m in months if m > pd.Timestamp(start)]

    frames: list[pd.DataFrame] = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        url = (
            f"{MX_HISTORICAL_URL}?symbol={CORRA_FUTURES_ROOT}"
            f"&from={chunk_start:%Y-%m-%d}&to={chunk_end:%Y-%m-%d}&dnld=1"
        )
        raw = _http_get(url)
        frame = pd.read_csv(io.BytesIO(raw))
        if frame.empty:
            continue
        missing = set(_MX_KEEP_COLUMNS) - set(frame.columns)
        if missing:
            raise RuntimeError(f"MX response for {chunk_start:%Y-%m} is missing columns: {sorted(missing)}")
        frames.append(frame.loc[:, list(_MX_KEEP_COLUMNS)].rename(columns=_MX_KEEP_COLUMNS))
        print(f"  {chunk_start:%Y-%m}: {len(frame)} rows")
        time.sleep(_REQUEST_PAUSE_SECONDS)

    if not frames:
        raise RuntimeError("MX returned no rows for the requested window.")
    out = pd.concat(frames, ignore_index=True)
    # Pre-2025 records occasionally carry a placeholder expiry ("0000-00-00").
    # Coerce rather than fail: the expiry column is provenance, never used by
    # the implied-probability maths. A bad trading date, by contrast, would
    # silently misdate a settlement, so those rows are dropped and reported.
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["expiry_date"] = pd.to_datetime(out["expiry_date"], errors="coerce")
    undated = int(out["date"].isna().sum())
    if undated:
        print(f"  dropped {undated} rows with an unparseable trading date")
        out = out[out["date"].notna()]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["expiry_date"] = out["expiry_date"].dt.strftime("%Y-%m-%d")
    return out.drop_duplicates(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def fetch_valet_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Download one Bank of Canada Valet series as ``date`` / ``value`` rows."""
    url = f"{VALET_OBSERVATIONS_URL.format(series_id=series_id)}?start_date={start}&end_date={end}"
    text = _http_get(url).decode("utf-8")
    # The Valet CSV prefixes the observation block with a metadata section
    # terminated by a blank line; the observation header starts with "date".
    lines = text.splitlines()
    header_index = next(
        i for i, line in enumerate(lines) if line.lower().startswith('"date"') or line.startswith("date")
    )
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    frame.columns = ["date", "value"]
    frame = frame.dropna(subset=["value"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["value"] = frame["value"].astype(float)
    return frame.sort_values("date").reset_index(drop=True)


def main() -> None:
    """Fetch all three raw files into ``a1_market_implied/data/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    # The default window starts at the beginning of COA trading so one run
    # reproduces the committed files, which the 2023-24 replication window in
    # turn_detection/ depends on.
    parser.add_argument("--start", default="2023-01-01", help="First date to fetch (default: 2023-01-01).")
    parser.add_argument("--end", default="2026-09-20", help="Last date to fetch (default: 2026-09-20).")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets: dict[Path, Callable[[], pd.DataFrame]] = {
        DATA_DIR / "coa_settlements.csv": lambda: fetch_coa_settlements(args.start, args.end),
        DATA_DIR / "corra_daily.csv": lambda: fetch_valet_series(CORRA_SERIES_ID, args.start, args.end),
        DATA_DIR / "target_rate_daily.csv": lambda: fetch_valet_series(TARGET_RATE_SERIES_ID, args.start, args.end),
    }
    for path, fetch in targets.items():
        if path.exists() and not args.force:
            print(f"{path.name}: exists, skipping (use --force to refresh)")
            continue
        print(f"{path.name}: fetching...")
        frame = fetch()
        frame.to_csv(path, index=False)
        print(f"{path.name}: wrote {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
