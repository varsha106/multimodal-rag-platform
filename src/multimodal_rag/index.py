"""Hybrid dense + BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from .embeddings import Embedder
from .models import Chunk, SearchResult
from .text import tokenize

RRF_K = 60


class HybridIndex:
    """In-memory hybrid index suitable for a local reference deployment.

    FAISS is used when installed. If it is unavailable, the class falls back
    to exact NumPy cosine search, which is appropriate for the bundled sample
    corpus and for unit tests. BM25 uses ``rank_bm25`` when installed and a
    small local implementation otherwise.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.embedder = embedder
        self._embeddings = (
            embedder.embed([chunk.text for chunk in self.chunks])
            if self.chunks
            else np.zeros((0, embedder.dimension), dtype=np.float32)
        )
        self._faiss_index = self._build_faiss_index(self._embeddings)
        self._bm25 = _Bm25Index([chunk.text for chunk in self.chunks])
        self.vector_backend = "faiss" if self._faiss_index is not None else "numpy-exact"

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def document_count(self) -> int:
        return len({chunk.document_id for chunk in self.chunks})

    def add_chunks(self, new_chunks: Iterable[Chunk]) -> None:
        """Add chunks and update both retrieval indexes.

        This is intentionally simple—BM25 is rebuilt because the local index
        is in-memory and small. A production deployment would use a managed
        index with incremental updates.
        """

        additions = list(new_chunks)
        if not additions:
            return
        new_vectors = self.embedder.embed([chunk.text for chunk in additions])
        self.chunks.extend(additions)
        self._embeddings = (
            np.vstack([self._embeddings, new_vectors])
            if len(self._embeddings)
            else new_vectors
        )
        if self._faiss_index is not None:
            self._faiss_index.add(new_vectors)
        self._bm25 = _Bm25Index([chunk.text for chunk in self.chunks])

    def search(self, query: str, *, top_k: int = 5, candidate_k: int = 24) -> list[SearchResult]:
        """Search with one query and fuse dense/BM25 rankings using RRF."""

        if not self.chunks:
            return []
        limit = min(candidate_k, len(self.chunks))
        dense_ranking = self._dense_search(query, limit)
        bm25_ranking = self._bm25_search(query, limit)

        fused: dict[int, SearchResult] = {}
        for rank, (index, score) in enumerate(dense_ranking, start=1):
            result = fused.setdefault(index, SearchResult(chunk=self.chunks[index], score=0.0))
            result.dense_score = float(score)
            result.rrf_score = (result.rrf_score or 0.0) + 1.0 / (RRF_K + rank)
            result.matched_queries.append(query)
        for rank, (index, score) in enumerate(bm25_ranking, start=1):
            result = fused.setdefault(index, SearchResult(chunk=self.chunks[index], score=0.0))
            result.bm25_score = float(score)
            result.rrf_score = (result.rrf_score or 0.0) + 1.0 / (RRF_K + rank)
            if query not in result.matched_queries:
                result.matched_queries.append(query)

        results = sorted(
            fused.values(),
            key=lambda item: (item.rrf_score or 0.0, item.dense_score or 0.0),
            reverse=True,
        )
        for result in results:
            result.score = result.rrf_score or 0.0
        return results[:top_k]

    def search_many(
        self, queries: Sequence[str], *, top_k: int = 5, candidate_k: int = 24
    ) -> list[SearchResult]:
        """Fuse rankings from several planned query formulations."""

        if len(queries) == 1:
            return self.search(queries[0], top_k=top_k, candidate_k=candidate_k)

        merged: dict[str, SearchResult] = {}
        fusion_scores: dict[str, float] = defaultdict(float)
        for query in queries:
            ranked = self.search(query, top_k=candidate_k, candidate_k=candidate_k)
            for rank, result in enumerate(ranked, start=1):
                chunk_id = result.chunk.chunk_id
                fusion_scores[chunk_id] += 1.0 / (RRF_K + rank)
                if chunk_id not in merged:
                    merged[chunk_id] = SearchResult(
                        chunk=result.chunk,
                        score=0.0,
                        dense_score=result.dense_score,
                        bm25_score=result.bm25_score,
                        rrf_score=result.rrf_score,
                        matched_queries=list(result.matched_queries),
                    )
                else:
                    current = merged[chunk_id]
                    current.dense_score = _max_optional(current.dense_score, result.dense_score)
                    current.bm25_score = _max_optional(current.bm25_score, result.bm25_score)
                    current.rrf_score = _max_optional(current.rrf_score, result.rrf_score)
                    for matched in result.matched_queries:
                        if matched not in current.matched_queries:
                            current.matched_queries.append(matched)

        results = sorted(merged.values(), key=lambda item: fusion_scores[item.chunk.chunk_id], reverse=True)
        for result in results:
            result.score = fusion_scores[result.chunk.chunk_id]
        return results[:top_k]

    def _dense_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_vector = self.embedder.embed([query])
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vector, limit)
            return [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0])
                if index >= 0
            ]
        scores = (self._embeddings @ query_vector[0]).astype(float)
        indices = np.argsort(-scores)[:limit]
        return [(int(index), float(scores[index])) for index in indices]

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = self._bm25.scores(query)
        indices = np.argsort(-scores)[:limit]
        return [(int(index), float(scores[index])) for index in indices]

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray):
        if not len(embeddings):
            return None
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index


class _Bm25Index:
    """BM25 backed by rank-bm25 when available, with a local fallback."""

    def __init__(self, texts: Sequence[str]) -> None:
        self._tokenized = [tokenize(text) for text in texts]
        self._external = None
        try:
            from rank_bm25 import BM25Okapi

            self._external = BM25Okapi(self._tokenized) if self._tokenized else None
        except ImportError:
            self._external = None

        document_count = len(self._tokenized)
        self._lengths = [len(tokens) for tokens in self._tokenized]
        self._average_length = (
            sum(self._lengths) / document_count if document_count else 0.0
        )
        self._term_frequencies: list[dict[str, int]] = []
        document_frequencies: dict[str, int] = defaultdict(int)
        for tokens in self._tokenized:
            frequencies: dict[str, int] = defaultdict(int)
            for token in tokens:
                frequencies[token] += 1
            self._term_frequencies.append(dict(frequencies))
            for token in frequencies:
                document_frequencies[token] += 1
        self._idf = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query: str) -> np.ndarray:
        if not self._tokenized:
            return np.zeros(0, dtype=float)
        query_tokens = tokenize(query)
        if self._external is not None:
            return np.asarray(self._external.get_scores(query_tokens), dtype=float)

        k1 = 1.5
        b = 0.75
        output = np.zeros(len(self._tokenized), dtype=float)
        for document_index, frequencies in enumerate(self._term_frequencies):
            length = self._lengths[document_index] or 1
            score = 0.0
            for token in set(query_tokens):
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (1.0 - b + b * length / self._average_length)
                score += self._idf.get(token, 0.0) * (frequency * (k1 + 1.0) / denominator)
            output[document_index] = score
        return output


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)
