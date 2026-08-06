"""Read-only access to pre-cached WTI news briefings, keyed by origin date.

Live ``search_web`` calls have the same reproducibility problem the
statistical anchor had before it was externalized (see
``analyst_agent/anchor_lookup.py`` and
``planning-docs/anchor-externalization-interview-notes.md``): two calls for
the same origin are not guaranteed to return the same result, which
contaminates mechanism-isolation comparisons that assume every variant sees
the same input. This module reads news briefings that were fetched once,
with a hard temporal cutoff, and committed to the repo — see
``scripts/cache_wti_curriculum_news.py``.

This module has no local imports (beyond the shared ``curriculum`` loader
helper) so it can be shared between the prompt builder and the config
factory without a cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aieng.forecasting.methods.agentic.curriculum import load_context_documents


_DEFAULT_CONTEXT_DIR = Path(__file__).parent.parent / "adaptive_agent" / "curriculum" / "context"


class NewsCacheSource:
    """Read-only view over pre-cached WTI news briefings.

    Parameters
    ----------
    context_dir : Path
        Directory of ``wti_news_<YYYY-MM-DD>.md`` files. Defaults to the
        cache built by ``scripts/cache_wti_curriculum_news.py``
        (``adaptive_agent/curriculum/context/``) — the same files back
        adaptive_agent's curriculum delivery and the anchored analyst
        variant's news briefing, since both just need "what was known as of
        this date," not anything agent-specific.
    """

    def __init__(self, context_dir: Path = _DEFAULT_CONTEXT_DIR) -> None:
        self._context_dir = context_dir

    def get(self, *, as_of: Any) -> str:
        """Look up the cached news briefing for one origin date.

        Parameters
        ----------
        as_of : Any
            Forecast origin date; stringified and truncated to ``YYYY-MM-DD``,
            matching the convention used throughout ``agent.py``.

        Returns
        -------
        str
            The cached briefing's markdown content.

        Raises
        ------
        KeyError
            If no cached file exists for the date. This is a deliberate
            fail-loud boundary, mirroring ``AnchorSource.get()``: silently
            falling back to a live ``search_web`` call would defeat the
            reproducibility this cache exists for.
        """
        date_key = str(as_of)[:10]
        docs = load_context_documents(self._context_dir, [date_key])
        if not docs:
            raise KeyError(f"No cached news briefing for as_of={date_key!r} in {self._context_dir}")
        _, content = docs[0]
        return content
