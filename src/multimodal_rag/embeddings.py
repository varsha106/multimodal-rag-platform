"""Embedding providers.

``SentenceTransformerEmbedder`` is the normal local provider. The hashing
provider is a deterministic, dependency-light fallback for CI, tests, and
air-gapped demos; it captures lexical similarity, not transformer semantics,
and the rest of the system labels it accordingly when it is selected.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dimension)`` float32 array of L2-normalized rows."""


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedding provider."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise RuntimeError(
                "sentence-transformers is not installed; install requirements.txt "
                "or select the hashing embedder for an offline smoke test"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:  # pragma: no cover - defensive for provider changes
            sample = self._model.encode(["dimension probe"], normalize_embeddings=True)
            dimension = int(sample.shape[1])
        self.dimension = int(dimension)
        self.name = f"sentence-transformer:{model_name}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic lexical hashing embeddings for offline tests and demos."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.name = "hashing-lexical-v1"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
            features.extend(
                (f"{left}__{right}", 0.65) for left, right in itertools.pairwise(tokens)
            )
            for feature, weight in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, column] += sign * weight * (1.0 + math.log1p(len(feature) % 5))
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


def build_embedder(kind: str, model_name: str) -> Embedder:
    """Build an embedder by name, with a safe fallback for ``auto`` mode."""

    normalized = kind.strip().lower()
    if normalized == "hashing":
        return HashingEmbedder()
    if normalized in {"sentence-transformer", "sentence_transformer", "st"}:
        return SentenceTransformerEmbedder(model_name)
    if normalized == "auto":
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:  # noqa: BLE001 - auto mode is explicitly a best-effort fallback
            return HashingEmbedder()
    raise ValueError(f"Unknown embedder kind: {kind}")
