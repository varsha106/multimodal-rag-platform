"""Application assembly shared by the CLI and API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import RagWorkflow
from .chunking import TextChunker
from .config import Settings
from .embeddings import build_embedder
from .index import HybridIndex
from .ingestion import load_directory
from .reranking import build_reranker


@dataclass
class RagService:
    settings: Settings
    index: HybridIndex
    workflow: RagWorkflow

    @property
    def document_count(self) -> int:
        return self.index.document_count

    @property
    def chunk_count(self) -> int:
        return len(self.index)


def build_service(settings: Settings | None = None, data_dir: str | Path | None = None) -> RagService:
    """Load documents, build the hybrid index, and assemble the workflow."""

    resolved = settings or Settings.from_env()
    if data_dir is not None:
        resolved = Settings(
            data_dir=Path(data_dir),
            chunk_size=resolved.chunk_size,
            chunk_overlap=resolved.chunk_overlap,
            embedder=resolved.embedder,
            embedding_model=resolved.embedding_model,
            reranker=resolved.reranker,
            reranker_model=resolved.reranker_model,
            candidate_k=resolved.candidate_k,
            top_k=resolved.top_k,
        )

    documents = load_directory(resolved.data_dir)
    chunker = TextChunker(resolved.chunk_size, resolved.chunk_overlap)
    chunks = chunker.split_documents(documents)
    embedder = build_embedder(resolved.embedder, resolved.embedding_model)
    index = HybridIndex(chunks, embedder)
    workflow = RagWorkflow(
        index,
        build_reranker(resolved.reranker, resolved.reranker_model),
        candidate_k=resolved.candidate_k,
        top_k=resolved.top_k,
    )
    return RagService(settings=resolved, index=index, workflow=workflow)
