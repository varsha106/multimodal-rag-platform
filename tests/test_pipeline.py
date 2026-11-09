from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_rag.agent import RagWorkflow
from multimodal_rag.chunking import TextChunker
from multimodal_rag.config import Settings
from multimodal_rag.embeddings import HashingEmbedder
from multimodal_rag.index import HybridIndex
from multimodal_rag.ingestion import documents_from_texts, load_directory
from multimodal_rag.reranking import LexicalReranker
from multimodal_rag.service import build_service


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        documents = documents_from_texts(
            [
                (
                    "Retrieval Design",
                    "memory://retrieval",
                    (
                        "Hybrid retrieval combines dense vector search with BM25 keyword "
                        "search. Reciprocal rank fusion merges the rankings, and each "
                        "answer cites the source passage used as evidence."
                    ),
                ),
                (
                    "Latency Guide",
                    "memory://latency",
                    (
                        "When latency rises, inspect retrieval, reranking, and generation "
                        "separately. Record timeouts by stage and check embedding backlogs."
                    ),
                ),
            ]
        )
        chunks = TextChunker(chunk_size=500, chunk_overlap=50).split_documents(documents)
        self.index = HybridIndex(chunks, HashingEmbedder())
        self.workflow = RagWorkflow(self.index, LexicalReranker(), candidate_k=10, top_k=3)

    def test_chunker_preserves_source_metadata(self) -> None:
        documents = documents_from_texts(
            [("A Title", "memory://one", "First sentence. Second sentence. " * 30)]
        )
        chunks = TextChunker(chunk_size=160, chunk_overlap=20).split_documents(documents)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source == "memory://one" for chunk in chunks))
        self.assertTrue(all(chunk.title == "A Title" for chunk in chunks))

    def test_hybrid_search_finds_exact_keyword_document(self) -> None:
        results = self.index.search("Which method uses reciprocal rank fusion?", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.title, "Retrieval Design")
        self.assertIsNotNone(results[0].bm25_score)

    def test_answer_is_grounded_and_cited(self) -> None:
        answer = self.workflow.ask("How does hybrid retrieval combine vector and keyword search?")
        self.assertTrue(answer.sufficient_evidence)
        self.assertTrue(answer.citations)
        self.assertIn("[1]", answer.text)
        self.assertTrue(any("dense vector" in result.chunk.text.lower() for result in answer.results))

    def test_bundled_sample_corpus_builds(self) -> None:
        settings = Settings(
            data_dir=ROOT / "data" / "sample_docs",
            embedder="hashing",
            reranker="lexical",
        )
        service = build_service(settings)
        self.assertGreaterEqual(service.document_count, 3)
        self.assertGreater(service.chunk_count, service.document_count)


class SampleCorpusTests(unittest.TestCase):
    def test_sample_directory_loads_supported_files(self) -> None:
        documents = load_directory(ROOT / "data" / "sample_docs")
        titles = {document.title for document in documents}
        self.assertIn("Enterprise Knowledge Platform Overview", titles)
        self.assertIn("Enterprise Search Operations Runbook", titles)


if __name__ == "__main__":
    unittest.main()
