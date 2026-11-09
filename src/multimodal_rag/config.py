"""Runtime configuration for the reference RAG service.

Configuration is intentionally small and environment-driven. No cloud
credentials are required for the local implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Settings used by the demo and FastAPI application."""

    data_dir: Path = Path("data/sample_docs")
    chunk_size: int = 900
    chunk_overlap: int = 140
    embedder: str = "auto"  # auto | sentence-transformer | hashing
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "lexical"  # lexical | cross-encoder | none
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 24
    top_k: int = 5

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``RAG_*`` environment variables."""

        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data/sample_docs")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "140")),
            embedder=os.getenv("RAG_EMBEDDER", "auto"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            reranker=os.getenv("RAG_RERANKER", "lexical"),
            reranker_model=os.getenv(
                "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", "24")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
        )
