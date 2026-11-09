"""Second-stage rerankers for hybrid-search candidates."""

from __future__ import annotations

from typing import Protocol

from .models import SearchResult
from .text import content_tokens


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return results in final presentation order."""


class NoOpReranker:
    name = "none"

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return list(results)


class LexicalReranker:
    """Transparent local reranker based on query-term coverage.

    This is not a neural cross-encoder. It is useful for deterministic tests
    and as a safe fallback when model downloads are unavailable.
    """

    name = "lexical-coverage-v1"

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_tokens = content_tokens(query)
        reranked: list[SearchResult] = []
        for result in results:
            text_tokens = content_tokens(result.chunk.text)
            title_tokens = content_tokens(result.chunk.title)
            if query_tokens:
                coverage = len(query_tokens & text_tokens) / len(query_tokens)
                title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
            else:
                coverage = 0.0
                title_coverage = 0.0
            phrase_bonus = 0.15 if query.lower() in result.chunk.text.lower() else 0.0
            score = coverage + 0.25 * title_coverage + phrase_bonus
            reranked.append(_copy_with_rerank_score(result, score))
        return sorted(
            reranked,
            key=lambda item: (item.rerank_score or 0.0, item.rrf_score or item.score),
            reverse=True,
        )


class CrossEncoderReranker:
    """Neural cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc
        self.model_name = model_name
        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = self._model.predict(pairs, show_progress_bar=False)
        reranked = [
            _copy_with_rerank_score(result, float(score))
            for result, score in zip(results, scores)
        ]
        return sorted(reranked, key=lambda item: item.rerank_score or 0.0, reverse=True)


def build_reranker(kind: str, model_name: str) -> Reranker:
    normalized = kind.strip().lower()
    if normalized == "none":
        return NoOpReranker()
    if normalized == "lexical":
        return LexicalReranker()
    if normalized in {"cross-encoder", "cross_encoder", "neural"}:
        return CrossEncoderReranker(model_name)
    raise ValueError(f"Unknown reranker kind: {kind}")


def _copy_with_rerank_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        chunk=result.chunk,
        score=score,
        dense_score=result.dense_score,
        bm25_score=result.bm25_score,
        rrf_score=result.rrf_score,
        rerank_score=score,
        matched_queries=list(result.matched_queries),
    )
