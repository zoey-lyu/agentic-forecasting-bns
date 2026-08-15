"""Tests for cutoff-aware BoC research evidence selection and formatting."""

from __future__ import annotations

from datetime import date, datetime

from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument
from boc_rate_decisions.research import format_research_evidence, latest_research_documents


def _doc(source: str, doc_id: str, published: date, text: str) -> ExtractedDocument:
    return ExtractedDocument(
        meta=DocumentMeta(
            source=source,
            doc_id=doc_id,
            publication_date=published,
            title=f"Title {doc_id}",
        ),
        text=text,
        page_count=1,
        n_chars=len(text),
        est_tokens=max(1, len(text) // 4),
        extracted_at=datetime(2026, 1, 1),
    )


class _CutoffContext:
    """Minimal context double that applies the production cutoff contract."""

    def __init__(self, documents: list[ExtractedDocument], as_of: date) -> None:
        self.documents = documents
        self.as_of = as_of

    def get_documents(self, source: str) -> list[ExtractedDocument]:
        return [
            doc
            for doc in self.documents
            if doc.meta.source == source and doc.meta.publication_date <= self.as_of
        ]


def test_latest_documents_merge_sources_after_cutoff_filtering() -> None:
    documents = [
        _doc("statements", "old", date(2024, 1, 24), "old"),
        _doc("surveys", "bos", date(2024, 4, 15), "survey"),
        _doc("statements", "latest", date(2024, 4, 10), "statement"),
        _doc("statements", "future", date(2024, 6, 5), "must stay hidden"),
    ]
    context = _CutoffContext(documents, date(2024, 5, 8))

    selected = latest_research_documents(  # type: ignore[arg-type]
        context, sources=("statements", "surveys"), max_documents=2
    )

    assert [doc.meta.doc_id for doc in selected] == ["latest", "bos"]
    assert all(doc.meta.doc_id != "future" for doc in selected)


def test_prompt_evidence_is_bounded_and_carries_provenance() -> None:
    context = _CutoffContext(
        [_doc("statements", "2024-04-10_en", date(2024, 4, 10), "abcdefghij")],
        date(2024, 5, 8),
    )

    evidence = format_research_evidence(  # type: ignore[arg-type]
        context, sources=("statements",), max_chars_per_document=5
    )

    assert evidence == [
        {
            "source": "statements",
            "document_id": "2024-04-10_en",
            "title": "Title 2024-04-10_en",
            "publication_date": "2024-04-10",
            "text": "abcde\n[truncated]",
        }
    ]


def test_zero_document_budget_returns_no_evidence() -> None:
    context = _CutoffContext(
        [_doc("statements", "one", date(2024, 4, 10), "text")],
        date(2024, 5, 8),
    )
    assert latest_research_documents(context, sources=("statements",), max_documents=0) == []  # type: ignore[arg-type]
