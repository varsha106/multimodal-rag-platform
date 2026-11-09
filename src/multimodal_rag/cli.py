"""Command-line demo for the local RAG platform."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .service import build_service

DEFAULT_QUESTIONS = [
    "How does the platform combine vector and keyword search?",
    "How are unsupported claims prevented in generated answers?",
    "What should an operator check when answer latency rises?",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the multimodal RAG demo")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sample_docs"))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic hashing embeddings and lexical reranking (no model download).",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "sentence-transformer", "hashing"],
        default="auto",
        help="Embedding provider. Ignored when --offline is set.",
    )
    parser.add_argument(
        "--reranker",
        choices=["lexical", "cross-encoder", "none"],
        default="lexical",
        help="Second-stage reranker. Ignored when --offline is set.",
    )
    parser.add_argument("--question", action="append", dest="questions", help="Ask a custom question; may be repeated.")
    args = parser.parse_args(argv)

    settings = Settings(
        data_dir=args.data_dir,
        embedder="hashing" if args.offline else args.embedder,
        reranker="lexical" if args.offline else args.reranker,
    )
    service = build_service(settings)

    print("Multimodal RAG Platform — local demo")
    print(f"Documents: {service.document_count} | Chunks: {service.chunk_count}")
    print(f"Embedder: {service.index.embedder.name} | Vector backend: {service.index.vector_backend}")
    print(f"Reranker: {service.workflow.reranker.name}")
    print()

    for question in args.questions or DEFAULT_QUESTIONS:
        answer = service.workflow.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer.text}")
        if answer.citations:
            print("Sources:")
            for citation in answer.citations:
                page = f", page {citation.page}" if citation.page else ""
                print(f"  {citation.marker} {citation.title}{page} — {citation.source}")
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
