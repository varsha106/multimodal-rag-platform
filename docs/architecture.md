# Architecture Notes

## Purpose

This repository implements the retrieval and grounding core of an enterprise multimodal RAG platform in a form that can be inspected, tested, and run locally. It mirrors the production system's architecture while replacing proprietary services and data with open-source components and synthetic documents.

## Request lifecycle

1. **Load** — `ingestion.py` reads PDF, Markdown, and text files. PDF text is extracted page by page; source, format, page count, and modality metadata are retained.
2. **Chunk** — `chunking.py` creates overlapping, retrieval-sized passages while preferring paragraph and sentence boundaries.
3. **Index** — `embeddings.py` produces normalized vectors; `index.py` stores them in FAISS (or NumPy for small/test corpora) and builds a BM25 index over the same chunks.
4. **Plan** — `agent.py` creates one or more query formulations. The local planner uses transparent concept expansions; a production workflow can add model-based rewriting while keeping the original query in the fusion set.
5. **Retrieve and fuse** — each formulation runs dense and BM25 search. Reciprocal-rank fusion combines rankings without mixing incompatible raw score scales.
6. **Rerank** — `reranking.py` applies either a neural cross-encoder or a deterministic lexical fallback to the candidate pool.
7. **Generate** — `generation.py` creates an extractive answer from selected sentences only and assigns citations at chunk level.
8. **Serve** — `api.py` exposes the workflow through FastAPI; `cli.py` offers the same path for terminal demos and smoke tests.

## Why hybrid retrieval?

Dense retrieval is useful when the user paraphrases a document. BM25 is useful when the query contains an exact product name, identifier, error string, or uncommon term. Neither method is uniformly better. Keeping both rankings visible also makes failures diagnosable: an operator can see whether a missing passage was absent from both candidate sets, lost during fusion, or dropped by reranking.

## Grounding contract

A generated answer should satisfy three rules:

- every factual sentence is supported by at least one retrieved chunk;
- every citation resolves to the exact chunk used as evidence;
- missing evidence produces an explicit insufficient-evidence response.

The local extractive generator enforces the contract mechanically. If the generator is replaced with a Bedrock-hosted LLM, the LLM should receive the same evidence pack, be required to cite chunk IDs, and be evaluated with retrieval relevance, answer relevance, faithfulness, and citation-precision checks before release.

## Production substitutions

| Boundary | Local implementation | Production direction |
| --- | --- | --- |
| Corpus storage | Files on disk | Governed document stores and data lake/warehouse sources |
| PDF/media extraction | `pypdf` text layer | OCR, table extraction, image captioning, and document-layout services |
| Vector index | FAISS in memory | Amazon OpenSearch vector fields with sharding and replication |
| Keyword index | `rank-bm25` in memory | OpenSearch BM25 in the same governed index |
| Embedding model | Local sentence-transformers model | Amazon Bedrock embeddings |
| Generation | Extractive composer | Amazon Bedrock LLM with citation validation |
| Orchestration | Local workflow + LangChain LCEL adapter | LangChain agents/tools with tracing and guardrails |
| Serving | Uvicorn container | Container platform behind an authenticated API gateway |
| Evaluation | Unit tests and manual sample questions | Offline golden sets plus online feedback and drift monitoring |

## Security considerations for a real deployment

- Apply source-document authorization **before** reranking and generation; never filter only the displayed citations after restricted text has influenced an answer.
- Treat uploaded documents and retrieved text as untrusted input. Prompt-injection text inside a document should not be able to change system instructions or tool permissions.
- Log query, document IDs, model versions, and citation IDs for auditability while minimizing retention of sensitive query text.
- Version indexes and models together so an answer can be reproduced against the corpus that produced it.
- Keep deletion and retention workflows in the ingestion pipeline so removed source documents also disappear from derived chunks and indexes.

## Scaling path

The compact in-memory design can evolve in stages:

1. persist chunk metadata and embeddings in OpenSearch;
2. move ingestion and embedding generation to an asynchronous job pipeline;
3. add incremental document versioning and deletion;
4. introduce a hosted cross-encoder with candidate-pool limits and latency budgets;
5. add golden-set evaluation to CI and index-promotion workflows;
6. separate search and generation services if their scaling profiles diverge.

Each step preserves the public contracts in `models.py`: `Document`, `Chunk`, `SearchResult`, `Citation`, and `Answer`.
