"""FastAPI service for hybrid search and grounded question answering."""

from __future__ import annotations

import os
from typing import Any

from .chunking import TextChunker
from .config import Settings
from .ingestion import documents_from_texts
from .models import SearchResult
from .service import RagService, build_service

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
    raise RuntimeError("FastAPI support requires fastapi and pydantic") from exc


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


def create_app(service: RagService | None = None) -> FastAPI:
    rag = service or build_service(Settings.from_env())
    app = FastAPI(
        title="Multimodal RAG Platform — Reference Implementation",
        version="0.1.0",
        description="Hybrid retrieval and grounded answers over local documents.",
    )
    app.state.rag = rag

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "documents": rag.document_count,
            "chunks": rag.chunk_count,
            "embedder": rag.index.embedder.name,
            "vector_backend": rag.index.vector_backend,
            "reranker": rag.workflow.reranker.name,
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return {
            "documents": rag.document_count,
            "chunks": rag.chunk_count,
            "data_dir": str(rag.settings.data_dir),
            "embedder": rag.index.embedder.name,
            "vector_backend": rag.index.vector_backend,
        }

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        results = rag.workflow.search(request.query, top_k=request.top_k)
        return {"query": request.query, "results": [_result_dict(result) for result in results]}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        answer = rag.workflow.ask(request.question, top_k=request.top_k)
        return answer.to_dict()

    @app.post("/documents", status_code=201)
    def add_document(request: DocumentRequest) -> dict[str, Any]:
        documents = documents_from_texts([(request.title, request.source, request.text)])
        chunker = TextChunker(rag.settings.chunk_size, rag.settings.chunk_overlap)
        chunks = chunker.split_documents(documents)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no chunks")
        rag.index.add_chunks(chunks)
        return {"document_id": documents[0].document_id, "chunks_added": len(chunks)}

    return app


def _result_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk.chunk_id,
        "document_id": result.chunk.document_id,
        "title": result.chunk.title,
        "source": result.chunk.source,
        "page": result.chunk.metadata.get("page"),
        "score": result.score,
        "dense_score": result.dense_score,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "rerank_score": result.rerank_score,
        "matched_queries": result.matched_queries,
        "text": result.chunk.text,
    }


app = create_app()


def run() -> None:
    """Console-script entry point for the API server."""

    import uvicorn

    uvicorn.run(
        "multimodal_rag.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
