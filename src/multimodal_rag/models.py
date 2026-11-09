"""Typed records passed between ingestion, retrieval, and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document before chunking.

    ``metadata`` preserves source-system fields such as page count, author, or
    modality. The local loaders populate the fields they can observe; they do
    not infer facts that are not present in the source file.
    """

    document_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of a document."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        page = self.metadata.get("page")
        suffix = f", p. {page}" if page else ""
        return f"{self.title}{suffix}"


@dataclass
class SearchResult:
    """A chunk plus the scores assigned by each retrieval stage."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    matched_queries: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    """A source reference attached to a generated answer."""

    marker: str
    title: str
    source: str
    chunk_id: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "title": self.title,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page": self.page,
        }


@dataclass(frozen=True)
class Answer:
    """A grounded answer and the evidence used to produce it."""

    question: str
    text: str
    citations: list[Citation]
    results: list[SearchResult]
    sufficient_evidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "sufficient_evidence": self.sufficient_evidence,
            "citations": [citation.to_dict() for citation in self.citations],
            "results": [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "title": result.chunk.title,
                    "source": result.chunk.source,
                    "score": result.score,
                    "dense_score": result.dense_score,
                    "bm25_score": result.bm25_score,
                    "rrf_score": result.rrf_score,
                    "rerank_score": result.rerank_score,
                    "text": result.chunk.text,
                }
                for result in self.results
            ],
        }
