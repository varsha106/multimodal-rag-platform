"""Agentic RAG workflow.

The workflow makes its stages explicit and inspectable:

1. plan and expand the user's query;
2. retrieve with dense + BM25 hybrid search for each formulation;
3. fuse the rankings;
4. rerank a candidate pool;
5. generate an answer only from cited evidence.

``build_langchain_chain`` exposes the same stages as a LangChain LCEL chain
when ``langchain-core`` is installed. The framework-independent workflow is
kept as the execution core so the service and tests do not depend on a
particular agent framework's changing tool APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .generation import GroundedAnswerGenerator
from .index import HybridIndex
from .models import Answer, SearchResult
from .reranking import Reranker
from .text import tokenize

_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("hybrid", "search", "retrieval"), ("dense vector", "BM25", "reciprocal rank fusion")),
    (("rag",), ("retrieval augmented generation", "grounded citations")),
    (("latency", "slow", "timeout"), ("p95", "response time", "retrieval timeout", "reranking")),
    (("citation", "citations", "grounded"), ("source attribution", "evidence", "unsupported claims")),
    (("evaluate", "evaluation", "quality"), ("relevance", "faithfulness", "answer correctness")),
    (("ingest", "ingestion", "indexing"), ("chunking", "embeddings", "document refresh")),
    (("access", "permission", "security"), ("authorization", "source permissions", "audit")),
)


@dataclass(frozen=True)
class QueryPlan:
    original: str
    formulations: list[str]


class QueryPlanner:
    """Deterministic query expansion for common enterprise-search concepts."""

    def plan(self, question: str) -> QueryPlan:
        tokens = set(tokenize(question))
        formulations = [question]
        additions: list[str] = []
        for triggers, expansion_terms in _EXPANSIONS:
            if any(trigger in tokens for trigger in triggers):
                additions.extend(expansion_terms)
        if additions:
            formulations.append(f"{question} {' '.join(dict.fromkeys(additions))}")
        return QueryPlan(original=question, formulations=formulations)


class RagWorkflow:
    """End-to-end question-answering workflow over a :class:`HybridIndex`."""

    def __init__(
        self,
        index: HybridIndex,
        reranker: Reranker,
        *,
        candidate_k: int = 24,
        top_k: int = 5,
        generator: GroundedAnswerGenerator | None = None,
    ) -> None:
        self.index = index
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.planner = QueryPlanner()
        self.generator = generator or GroundedAnswerGenerator()

    def search(self, question: str, *, top_k: int | None = None) -> list[SearchResult]:
        plan = self.planner.plan(question)
        return self._retrieve(plan.formulations, top_k=top_k or self.top_k)

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        final_k = top_k or self.top_k
        results = self.search(question, top_k=final_k)
        return self.generator.generate(question, results)

    def _retrieve(self, formulations: list[str], *, top_k: int) -> list[SearchResult]:
        candidate_pool = self.index.search_many(
            formulations,
            top_k=max(self.candidate_k, top_k),
            candidate_k=self.candidate_k,
        )
        if not candidate_pool:
            return []
        query_for_rerank = formulations[0]
        reranked = self.reranker.rerank(query_for_rerank, candidate_pool)
        return reranked[:top_k]

    def build_langchain_chain(self) -> Any:
        """Return an LCEL chain that mirrors :meth:`ask`.

        Raises:
            RuntimeError: if ``langchain-core`` is not installed.
        """

        try:
            from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise RuntimeError("LangChain support requires the langchain package") from exc

        plan = RunnableLambda(lambda question: {"question": question, "plan": self.planner.plan(question)})
        retrieve = RunnableLambda(
            lambda state: {
                **state,
                "results": self._retrieve(state["plan"].formulations, top_k=self.top_k),
            }
        )
        answer = RunnableLambda(
            lambda state: self.generator.generate(state["question"], state["results"])
        )
        return RunnablePassthrough() | plan | retrieve | answer
