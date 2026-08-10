"""Pre-cache WTI news briefings for the 2025 backtest and 2026 eval windows.

Generates one markdown file per weekly origin (every Monday in the given date
range) by calling the proxy-backed web search with a temporal cutoff, and
writes the results to:

    implementations/energy_oil_forecasting/adaptive_agent/curriculum/context/

Files are named ``wti_news_<YYYY-MM-DD>.md``, keyed by the **origin** date (not
the search cutoff — see below). Existing files are skipped unless ``--force``
is passed.

Usage::

    uv run python scripts/cache_wti_curriculum_news.py

    # Custom date range (both energy_oil_backtest and energy_oil_eval need a run):
    uv run python scripts/cache_wti_curriculum_news.py --start 2025-01-01 --end 2025-12-31
    uv run python scripts/cache_wti_curriculum_news.py --start 2026-02-02 --end 2026-06-01

    # Overwrite existing files (the only way to repair a corrupted one):
    uv run python scripts/cache_wti_curriculum_news.py --force

    # Dry-run (show dates without fetching):
    uv run python scripts/cache_wti_curriculum_news.py --dry-run

Design notes
------------
See ``planning-docs/news-cache-rebuild-plan.md`` for the audit that motivated
this script's structure. In short:

- The search **cutoff** is the origin minus one business day, so the news and
  the price data (which stops at the previous close) share one information
  set. The **filename** stays keyed to the origin so ``NewsCacheSource.get``
  is unchanged downstream.
- The instruction/query ask for reporting from the seven days immediately
  preceding the cutoff (soft constraint — nothing here enforces it in code;
  see the plan's section 4.2 for why a hard floor was rejected).
- Before writing, each briefing must clear :func:`_validate_briefing`: a
  length floor, no chain-of-thought markers, no mid-sentence start, and no
  "current price" claim implausibly far outside the real trailing range.
  Failing that gate retries the search (up to :data:`_MAX_VALIDATION_ATTEMPTS`
  times) rather than writing a rejected briefing to disk.
- The leakage verifier (independent of the gate above, inside
  ``_build_search_tool`` itself) now runs on a different model/provider than
  the searcher — see the ``verifier_model`` override below — closing the
  independence bug documented in the plan's section 4.7.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "aieng-forecasting"))

# Load .env if present (for OPENAI_BASE_URL / OPENAI_API_KEY)
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

import pandas as pd
from aieng.forecasting.methods.agentic.agent_factory import (
    ContextRetrievalConfig,
    _apply_removals,
    _build_search_tool,
    _format_search_result,
    _search_once,
    _verify_no_leakage,
)
from energy_oil_forecasting.data import build_wti_service


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_OUTPUT_DIR = _REPO_ROOT / "implementations" / "energy_oil_forecasting" / "adaptive_agent" / "curriculum" / "context"

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

_SEARCH_QUERY = "WTI crude oil price market conditions supply demand OPEC outlook"

#: Days of reporting the query/instruction ask the model to stay within,
#: counting back from the cutoff. Soft constraint only (plan section 4.2) —
#: covers the most recent weekly EIA inventory report and keeps months-old
#: reporting from being presented as current.
_LOOKBACK_DAYS = 7

_SEARCH_INSTRUCTION = """\
You are a commodity market analyst reconstructing the information environment
at a specific historical date. The cutoff date is a hard constraint: you must
treat it as if you are operating on that date and have no knowledge of anything
that occurred after it. Do not reference, imply, or hint at events, prices,
decisions, or outcomes that were not yet public as of the cutoff.

Search for WTI crude oil market conditions publicly known strictly before the
cutoff date. Prefer sources reporting on developments from the seven days
immediately preceding the cutoff; do not present older reporting as if it
describes current market conditions. Summarise in 3-5 concise paragraphs
covering: price level and recent trend, OPEC+ production decisions,
geopolitical supply risks, demand outlook, and any notable analyst forecasts.
Use only sources dated before the cutoff. If a source is undated or ambiguous,
exclude it.

When sources disagree -- competing demand-growth forecasts, conflicting
price targets, differing reads on an OPEC+ decision -- report each source's
specific figure and who it is attributed to (e.g. "the IEA projects demand
growth of 700,000 bpd for 2025, versus OPEC's own forecast of 1.3 million
bpd"). Do not collapse a disagreement into a vague characterization of it
("agencies have divided views", "the outlook remains polarized", "estimates
vary widely") without also giving the actual competing numbers -- a reader
deciding what to believe needs the figures, not just the fact that they
differ.

CRITICAL: Do not include any information from after the cutoff date, even if
you believe it to be relevant context. The purpose of this summary is to
reconstruct what a market analyst would have known at that exact moment.

CRITICAL: Return only the summary itself. Do not narrate your search process,
restate these instructions, or think out loud about what to include — the
output is read verbatim as the finished briefing, so any deliberation in it
would be read as market context.\
"""

#: A different provider than search_model (gemini-3.5-flash), so the verifier
#: does not share the searcher's training lineage / knowledge-attribution
#: blind spot. Not a reasoning-tier model on this proxy, so it does not need
#: the reasoning_effort/extra_body workaround documented in
#: planning-docs/vector-llm-proxy.md — if a thinking-capable model is ever
#: substituted here, that workaround (plus a raised max_tokens) applies.
_VERIFIER_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _mondays_in_range(start: date, end: date) -> list[date]:
    """Return every Monday between start and end (inclusive)."""
    # Advance to the first Monday on or after start
    first = start + timedelta(days=(7 - start.weekday()) % 7)
    result: list[date] = []
    current = first
    while current <= end:
        result.append(current)
        current += timedelta(weeks=1)
    return result


def _prev_business_day(d: date) -> date:
    """The previous Mon-Fri calendar day.

    Not exchange-calendar aware (matches pandas' ``'B'`` frequency, which is
    what the rest of this repo's specs use for horizons) -- a market holiday
    the day before ``d`` is not accounted for. Origins in this project are
    always Mondays, for which this is exactly "the preceding Friday".
    """
    if d.weekday() == 0:  # Monday -> Friday
        return d - timedelta(days=3)
    if d.weekday() == 6:  # Sunday -> Friday
        return d - timedelta(days=2)
    return d - timedelta(days=1)


def _build_query(cutoff: date) -> str:
    """Append the lookback window to the base query, as a per-call date range."""
    lower_bound = cutoff - timedelta(days=_LOOKBACK_DAYS)
    return f"{_SEARCH_QUERY} (reporting dated between {lower_bound.isoformat()} and {cutoff.isoformat()})"


# ---------------------------------------------------------------------------
# Sanity gate (plan section 4.4) -- reject a briefing before it is written
# ---------------------------------------------------------------------------

_MIN_LENGTH = 300  # chars; real briefings run roughly 1500-2500 (plan section 1.4)

#: Deliberation markers observed in the five corrupted files from the original
#: audit (plan section 1.1). A briefing containing any of these is the model
#: talking to itself about the task, not a summary of it.
_COT_MARKERS = ("Let me", "Wait,", "The prompt asks", "I should")

_PRICE_RE = re.compile(r"\$(\d{2,3}(?:\.\d{1,2})?)\b")

#: Phrases that describe the *current* price level, as opposed to a
#: forward-looking quotation (EIA outlook projections, analyst targets) that a
#: bare dollar-figure regex cannot distinguish from a claim about now -- see
#: plan section 2 for why that distinction was necessary.
_CURRENT_CONTEXT_RE = re.compile(
    r"\b(currently|today|right now|as of (?:this|today)|trading (?:at|around|near)|"
    r"closed at|settled at|is at|stands at)\b",
    re.IGNORECASE,
)


_TERMINAL_PUNCT = (".", "!", "?", '."', '?"', '!"', ".)", "*")

#: A paragraph this short is a topic sentence with no elaboration ever
#: added -- e.g. "The demand outlook has become highly polarized among major
#: forecasting agencies." and nothing else. Observed directly across several
#: 2026-window files: the *whole-document* length floor doesn't catch this
#: because 3-4 such stub paragraphs can sum past it while none individually
#: says anything a forecaster could use.
_MIN_PARAGRAPH = 120
_BARE_HEADER_RE = re.compile(r"^\*\*[^*]+\*\*$")
_LEADING_HEADER_RE = re.compile(r"^\*\*[^*]+\*\*\s*\n?")


def _body_before_sources(text: str) -> str:
    """Strip the ``\\n\\nSources:\\n...`` footer :func:`_format_result` appends."""
    idx = text.find("\n\nSources:")
    return text if idx == -1 else text[:idx]


def _paragraph_defects(body: str) -> list[str]:
    """Per-paragraph checks the whole-document checks in :func:`_format_defects` miss.

    A document can clear the whole-document length floor and still end with
    proper punctuation while containing a mid-document paragraph that's
    truncated, a bare markdown header with nothing under it, or a stub
    topic-sentence paragraph that never got its supporting detail -- all
    observed directly on real 2026-window output. Checking each paragraph
    independently, not just the document's overall length and final
    character, is what catches these.
    """
    defects: list[str] = []
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        if _BARE_HEADER_RE.match(para):
            defects.append(f"paragraph {i} is a bare markdown header with no content: {para!r}")
            continue
        text = _LEADING_HEADER_RE.sub("", para).strip()
        if not text:
            defects.append(f"paragraph {i}'s header has no content beneath it: {para!r}")
            continue
        if len(text) < _MIN_PARAGRAPH:
            defects.append(f"paragraph {i} is a stub ({len(text)} chars) with no elaboration: {text!r}")
        if not text.endswith(_TERMINAL_PUNCT):
            defects.append(f"paragraph {i} does not end with terminal punctuation: ...{text[-60:]!r}")
    return defects


def _format_defects(text: str) -> list[str]:
    """Length floor, mid-sentence start/end, chain-of-thought markers, and per-paragraph checks.

    Every check here operates on ``body`` -- the text *before* the
    ``Sources:`` footer -- never on ``text`` as a whole. A real bug this
    caught after the fact: checking ``len(text)`` let a completely empty
    body (post-removal, nothing survived) sail past the length floor purely
    because the appended grounding-redirect URLs are individually ~150-200
    chars each, so five of them alone clear 300 chars with zero actual
    content. The floor exists to catch exactly that case, so it must be
    measured on the body alone.
    """
    defects: list[str] = []
    body = _body_before_sources(text).strip()
    if len(body) < _MIN_LENGTH:
        defects.append(f"too short (body is {len(body)} chars, floor is {_MIN_LENGTH})")
    if body and (body[0].islower() or body[0] in ",;:"):
        defects.append("begins lower-case / mid-sentence")
    if not body.endswith(_TERMINAL_PUNCT):
        defects.append("does not end with terminal punctuation (likely truncated mid-sentence, or body is empty)")
    for marker in _COT_MARKERS:
        if marker in text:
            defects.append(f"chain-of-thought marker present: {marker!r}")
    defects.extend(_paragraph_defects(body))
    return defects


def _implausible_current_prices(text: str, price_range: tuple[float, float]) -> list[str]:
    """Dollar figures near 'current price' language, outside the real trailing range."""
    lo, hi = price_range
    hits: list[str] = []
    for m in _PRICE_RE.finditer(text):
        window = text[max(0, m.start() - 40) : m.start()]
        if _CURRENT_CONTEXT_RE.search(window):
            value = float(m.group(1))
            if not (lo * 0.85 <= value <= hi * 1.15):
                hits.append(m.group(0))
    return hits


def _validate_briefing(text: str, price_range: tuple[float, float]) -> list[str]:
    """Empty list means the briefing clears the gate; otherwise, why it doesn't."""
    defects = _format_defects(text)
    defects += [f"implausible current price near {h!r}" for h in _implausible_current_prices(text, price_range)]
    return defects


def _load_close_series() -> pd.Series:
    """Full WTI close history, for the sanity gate's trailing-range check."""
    service = build_wti_service()
    raw = service.get_series("wti_crude_oil_price", as_of=datetime.now())
    return pd.Series(raw["value"].to_numpy(), index=pd.to_datetime(raw["timestamp"]).to_numpy()).sort_index()


def _trailing_range(series: pd.Series, as_of: date, *, window: int = 60) -> tuple[float, float]:
    """Min/max close over the ``window`` most recent trading days at or before ``as_of``."""
    window_vals = series[series.index <= pd.Timestamp(as_of)].iloc[-window:]
    return float(window_vals.min()), float(window_vals.max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_MAX_VALIDATION_ATTEMPTS = 3
_MAX_FALLBACK_ATTEMPTS = 3
#: Above this many removals, the surviving text's *coherence* can't be
#: trusted even though no individual span was confabulated -- observed
#: directly on a real run: 1-2 removals left clean prose, 8 removals left
#: dangling clause fragments ("In thin holiday trading leading up to
#: Christmas,") and orphaned back-references ("these acute geopolitical
#: disruptions") that no punctuation-level tidy-up can repair, because the
#: break is grammatical/semantic, not textual debris.
_MAX_FALLBACK_REMOVALS = 3


async def _best_effort_fallback(
    config: ContextRetrievalConfig,
    query: str,
    cutoff: date,
    *,
    price_range: tuple[float, float],
    openai_base_url: str,
    openai_api_key: str | None,
) -> tuple[str, str] | None:
    """Up to :data:`_MAX_FALLBACK_ATTEMPTS` more search+verify rounds, accepting
    the verifier's removals as-is once they're few enough to trust.

    Only reached after :data:`_MAX_VALIDATION_ATTEMPTS` full rounds through
    ``search_web`` (which requires ``clean=True`` *and* confidence above its
    threshold) have failed. In practice this happens on weeks with narrow,
    specific, recent claims (an exact tariff percentage, a named interception
    incident) that a memory-only independent verifier (no search tool of its
    own -- see the plan's section 4.7) can flag and remove but never reach
    full confidence about, even across repeated attempts with fresh searches.

    This still runs the same independent verifier and still refuses a
    confabulated span (a quote absent from its own input) -- that check is
    what makes trusting a partial-confidence verdict acceptable here: every
    removal is a verbatim, audited span from real search output, not
    something the verifier invented. What's relaxed is only the requirement
    that it be *fully* confident nothing else remains -- and even that is
    bounded by :data:`_MAX_FALLBACK_REMOVALS` and the same format gate every
    other path goes through, so a heavily-redacted, incoherent survivor is
    retried (fresh search -- content differs per call) rather than accepted.

    Returns
    -------
    tuple of (str, str) or None
        ``(content, note)`` from the first attempt that isn't confabulated,
        isn't over the removal cap, and clears :func:`_validate_briefing`;
        ``None`` if no attempt within the budget qualifies.

    TODO
    ----
    A search-capable *independent* (non-Gemini) verifier would likely resolve
    these cases properly instead of working around them -- investigated and
    deferred, not implemented; see planning-docs/news-cache-rebuild-plan.md
    section 4.8.
    """
    user_content = query + f"\n\nOnly include and cite information published strictly before {cutoff.isoformat()}."
    for attempt in range(1, _MAX_FALLBACK_ATTEMPTS + 1):
        content, sources = await _search_once(
            config, user_content, openai_base_url=openai_base_url, openai_api_key=openai_api_key
        )
        verdict = await _verify_no_leakage(
            text=content,
            query=query,
            cutoff_date=cutoff.isoformat(),
            verifier_model=config.verifier_model,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
        )
        filtered, confabulated = _apply_removals(content, verdict.removals)
        n_removed = len(verdict.removals)
        if confabulated is not None:
            print(f"    fallback attempt {attempt}/{_MAX_FALLBACK_ATTEMPTS}: confabulated span, skipping")
            continue
        if n_removed > _MAX_FALLBACK_REMOVALS:
            print(
                f"    fallback attempt {attempt}/{_MAX_FALLBACK_ATTEMPTS}: "
                f"{n_removed} removals exceeds the coherence cap ({_MAX_FALLBACK_REMOVALS}), skipping"
            )
            continue
        result = _format_search_result(filtered, sources)
        defects = _validate_briefing(result, price_range)
        if defects:
            print(f"    fallback attempt {attempt}/{_MAX_FALLBACK_ATTEMPTS}: failed gate: {'; '.join(defects)}")
            continue
        note = (
            f"best-effort fallback: independent verifier removed {n_removed} claim(s) but reported "
            f"clean={verdict.clean} confidence={verdict.confidence} (below the strict accept threshold) after "
            f"{_MAX_VALIDATION_ATTEMPTS} fully-verified attempts failed; removals were applied and accepted anyway "
            f"since none was a confabulated span and the count was within the coherence cap"
        )
        return result, note
    return None


async def _fetch_and_save(
    search_web: object,
    query_date: date,
    output_dir: Path,
    *,
    price_range: tuple[float, float],
    config: ContextRetrievalConfig,
    openai_base_url: str,
    openai_api_key: str | None,
    dry_run: bool = False,
    force: bool = False,
) -> str:
    """Fetch news for one origin and write to file.  Returns status string."""
    filename = output_dir / f"wti_news_{query_date}.md"
    if filename.exists() and not force:
        return f"  SKIP  {query_date} — {filename.name} already exists"

    if dry_run:
        return f"  DRY   {query_date} — would write {filename.name}"

    cutoff = _prev_business_day(query_date)
    query = _build_query(cutoff)

    content = ""
    defects: list[str] = ["not yet attempted"]
    for attempt in range(1, _MAX_VALIDATION_ATTEMPTS + 1):
        content = await search_web(query, cutoff_date=cutoff.isoformat())  # type: ignore[operator]
        defects = _validate_briefing(content, price_range)
        if not defects:
            break
        print(f"  RETRY {query_date} — attempt {attempt}/{_MAX_VALIDATION_ATTEMPTS} failed: {'; '.join(defects)}")

    fallback_note = ""
    if defects:
        print(f"  FALLBACK {query_date} — strict gate never passed, trying best-effort (up to {_MAX_FALLBACK_ATTEMPTS} attempts)")
        result = await _best_effort_fallback(
            config, query, cutoff, price_range=price_range, openai_base_url=openai_base_url, openai_api_key=openai_api_key
        )
        if result is None:
            return (
                f"  FAIL  {query_date} — gate never passed after {_MAX_VALIDATION_ATTEMPTS} attempts, "
                f"and no fallback attempt qualified either"
            )
        content, fallback_note = result

    header = (
        f"# WTI Market Context — {query_date}\n\n"
        f"*Pre-cached by `scripts/cache_wti_curriculum_news.py` "
        f"with search cutoff {cutoff} (origin {query_date} minus one business day, "
        f"{_LOOKBACK_DAYS}-day lookback).*\n"
        + (f"\n*Note: {fallback_note}.*\n" if fallback_note else "")
        + "\n---\n\n"
    )
    filename.write_text(header + content, encoding="utf-8")
    status = "OK*" if fallback_note else "OK"
    return f"  {status:<5} {query_date} — wrote {filename.name} ({len(content)} chars)"


async def main(
    start: date,
    end: date,
    *,
    output_dir: Path = _OUTPUT_DIR,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_base_url and not dry_run:
        print(
            "ERROR: OPENAI_BASE_URL is not set. Export it or add it to your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = ContextRetrievalConfig(
        enabled=True,
        instruction=_SEARCH_INSTRUCTION,
        search_model="gemini-3.5-flash",
        verifier_model=_VERIFIER_MODEL,
        enforce_cutoff=True,
        # Defence-in-depth against mid-sentence truncation (observed on a real
        # dry run at the 4096 default): the visible summary is well under this,
        # so the extra headroom is free for briefings that don't need it.
        max_output_tokens=8192,
    )
    search_web = _build_search_tool(
        config,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dates = _mondays_in_range(start, end)
    close_series = None if dry_run else _load_close_series()

    print(f"Date range: {start} → {end} ({len(dates)} Mondays)")
    print(f"Output dir: {output_dir}")
    print(f"Verifier model: {config.verifier_model} (search model: {config.search_model})")
    if dry_run:
        print("DRY RUN — no files will be written.\n")
    else:
        print()

    for d in dates:
        price_range = (0.0, 0.0) if close_series is None else _trailing_range(close_series, d)
        try:
            status = await _fetch_and_save(
                search_web,
                d,
                output_dir,
                price_range=price_range,
                config=config,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                dry_run=dry_run,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001 -- a proxy/network error on one date
            # must not abort the whole batch. Observed directly: the proxy
            # occasionally hangs for ~180s before finally raising a Timeout,
            # and previously that killed the entire process, discarding
            # every date already skipped/completed and forcing a full
            # restart. Catching here means the batch just moves on to the
            # next date; re-running without --force retries only this one.
            status = f"  ERROR {d} — {type(exc).__name__}: {exc}"
        print(status)
        if not dry_run:
            # Small delay to avoid proxy rate limits
            await asyncio.sleep(1.5)

    print(f"\nDone. {len(dates)} dates processed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--start",
        default="2025-01-01",
        help="Start date (YYYY-MM-DD). Default: 2025-01-01",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="End date (YYYY-MM-DD). Default: 2025-12-31",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping them. The only way to repair a corrupted briefing.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_OUTPUT_DIR),
        help=(
            "Directory to write briefings to. Default is the live NewsCacheSource path "
            f"({_OUTPUT_DIR}); point this elsewhere (e.g. a 'new_context' sibling dir) to "
            "regenerate for review without touching what every current experiment reads."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List dates that would be fetched without making any API calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        main(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            force=args.force,
        )
    )
