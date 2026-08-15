"""Cutoff-aware formatting of Bank of Canada research evidence.

Numeric and document cutoff enforcement belongs to :class:`ForecastContext`.
This module handles the use-case-specific step after retrieval: selecting the
most recent documents across configured sources and rendering a bounded,
provenance-rich block for the forecasting prompt.
"""

from __future__ import annotations

from collections.abc import Sequence

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.documents.models import ExtractedDocument


DEFAULT_RESEARCH_SOURCES: tuple[str, ...] = ("boc_press_releases",)
"""Document sources currently populated by the BoC acquisition scripts."""


def latest_research_documents(
    context: ForecastContext,
    *,
    sources: Sequence[str] = DEFAULT_RESEARCH_SOURCES,
    max_documents: int = 3,
) -> list[ExtractedDocument]:
    """Return the newest cutoff-visible documents across ``sources``.

    ``ForecastContext.get_documents`` performs the temporal filtering. This
    function only merges sources, sorts deterministically, and applies the
    document-count budget.
    """
    if max_documents < 0:
        raise ValueError("max_documents must be non-negative")
    documents = [doc for source in sources for doc in context.get_documents(source)]
    documents.sort(key=lambda doc: (doc.meta.publication_date, doc.meta.source, doc.meta.doc_id))
    return documents[-max_documents:] if max_documents else []


def format_research_evidence(
    context: ForecastContext,
    *,
    sources: Sequence[str] = DEFAULT_RESEARCH_SOURCES,
    max_documents: int = 3,
    max_chars_per_document: int = 6_000,
) -> list[dict[str, str]]:
    """Build structured, prompt-ready evidence with explicit provenance.

    Text is truncated per document to keep prompt cost predictable. The return
    value is JSON-serializable and is deliberately kept separate from the
    forecaster so it can later combine statements, MPRs, surveys, and speeches.
    """
    if max_chars_per_document <= 0:
        raise ValueError("max_chars_per_document must be positive")

    evidence: list[dict[str, str]] = []
    for doc in latest_research_documents(context, sources=sources, max_documents=max_documents):
        text = doc.text.strip()
        if len(text) > max_chars_per_document:
            text = text[:max_chars_per_document].rstrip() + "\n[truncated]"
        evidence.append(
            {
                "source": doc.meta.source,
                "document_id": doc.meta.doc_id,
                "title": doc.meta.title or doc.meta.doc_id,
                "publication_date": doc.meta.publication_date.isoformat(),
                "text": text,
            }
        )
    return evidence


__all__ = ["DEFAULT_RESEARCH_SOURCES", "format_research_evidence", "latest_research_documents"]
