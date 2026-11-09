# Multimodal RAG Platform — Reference Implementation

A production-style, open-source reference implementation of the retrieval architecture behind Varsha Puli's enterprise multimodal RAG work: document ingestion, chunking, embeddings, **hybrid dense + BM25 retrieval**, reciprocal-rank fusion, reranking, an agentic query workflow, grounded citation-backed answers, and a FastAPI service.

This repository is deliberately honest about scope. It is a faithful local implementation of the architecture, not a copy of proprietary code or internal documents. The bundled documents are small synthetic samples, and no production corpus, credentials, or performance claims are included.

## What it demonstrates

- PDF, Markdown, and plain-text ingestion
- Paragraph-aware chunking with overlap and source/page metadata
- Local embeddings with `sentence-transformers`
- Dense vector search with FAISS (or exact NumPy search as a portable fallback)
- BM25 keyword retrieval with `rank-bm25`
- Reciprocal-rank fusion (RRF) across dense, BM25, and multiple query formulations
- Optional neural cross-encoder reranking, plus a deterministic lexical reranker for tests and offline use
- A LangChain-compatible agentic workflow with explicit planning, retrieval, reranking, and answer stages
- Extractive grounded answers: every answer sentence comes from a retrieved passage and carries a citation
- FastAPI endpoints for health, corpus statistics, search, question answering, and adding a document
- Docker packaging and unit/API tests

## Architecture

```text
PDF / Markdown / Text
        │
        ▼
 Document loaders ──► metadata + normalized text
        │
        ▼
 Paragraph-aware chunker ──► overlapping chunks
        │
        ├──────────────► sentence-transformer embeddings ──► FAISS dense index
        │
        └──────────────► tokenized corpus ─────────────────► BM25 index
                                                                  │
User question                                                     │
        │                                                         │
        ▼                                                         │
 Query planner / expansion ──► hybrid search ◄────────────────────┘
        │
        ▼
 Reciprocal-rank fusion ──► cross-encoder or lexical reranking
        │
        ▼
 Evidence pack ──► grounded extractive answer ──► citations
        │
        ▼
 FastAPI / CLI
```

See [`docs/architecture.md`](docs/architecture.md) for module responsibilities, the request lifecycle, and production extension points.

## Scope and production mapping

The local code uses open-source components that can run on a laptop. The production architecture uses managed equivalents at the same interfaces:

| Concern | Local reference implementation | Production equivalent |
| --- | --- | --- |
| Document text extraction | `pypdf`, text/Markdown loaders | Managed document pipeline plus OCR, table, and image extractors |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Amazon Bedrock embedding model |
| Dense retrieval | FAISS `IndexFlatIP` | Amazon OpenSearch vector index |
| Keyword retrieval | `rank-bm25` | OpenSearch BM25 |
| Fusion | Local reciprocal-rank fusion | Hybrid query/fusion layer over OpenSearch results |
| Reranking | sentence-transformers cross-encoder or lexical fallback | Hosted or managed cross-encoder reranker |
| Agentic workflow | Explicit local workflow + LangChain LCEL adapter | LangChain workflows with enterprise tools |
| Answer generation | Extractive, citation-constrained local generator | Amazon Bedrock LLM constrained to the same evidence/citation contract |
| API | FastAPI + Uvicorn | FastAPI behind the organization's API platform |
| Packaging | Docker | Container deployment on AWS |

### Multimodal scope

The bundled implementation indexes text extracted from PDFs and text files. `Document` and `Chunk` records preserve format, source, page, and modality metadata so OCR output, extracted tables, and image captions can be added as additional evidence producers without changing retrieval or citation interfaces. It does not claim to perform image understanding in the local demo.

## Quickstart

### 1. Create an environment and install

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The first non-offline run may download the embedding model from Hugging Face. No AWS account or API key is needed.

### 2. Run the offline smoke demo

This command uses deterministic hashing embeddings and lexical reranking. It is the fastest way to verify the complete pipeline without a model download:

```bash
python scripts/demo.py --offline
```

Expected shape of the output:

```text
Multimodal RAG Platform — local demo
Documents: 4 | Chunks: 7
Embedder: hashing-lexical-v1 | Vector backend: faiss (or numpy-exact if FAISS is unavailable)
Reranker: lexical-coverage-v1

Q: How does the platform combine vector and keyword search?
A: Retrieval combines dense vector search with BM25 keyword search. [1]
   ...
Sources:
  [1] Enterprise Knowledge Platform Overview — data/sample_docs/platform_overview.txt
```

The hashing embedder is a lexical fallback for CI and air-gapped demos; use the default mode for transformer embeddings.

### 3. Run with local transformer embeddings

```bash
python scripts/demo.py --question "How are unsupported claims prevented in generated answers?"
```

Enable the neural cross-encoder with:

```bash
python scripts/demo.py --reranker cross-encoder \
  --question "What should an operator check when answer latency rises?"
```

### 4. Start the API

```bash
uvicorn multimodal_rag.api:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Search:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"reciprocal rank fusion","top_k":3}'
```

Ask a grounded question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does the platform combine vector and keyword search?","top_k":5}'
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the server is running.

### 5. Run with Docker

```bash
docker build -t multimodal-rag-platform .
docker run --rm -p 8000:8000 multimodal-rag-platform
```

Then open `http://127.0.0.1:8000/health`.

## Sample corpus

`data/sample_docs/` contains four small, synthetic documents:

- `platform_overview.txt` — ingestion, hybrid retrieval, fusion, reranking, and grounding
- `operations_runbook.md` — latency triage, refresh, and quality checks
- `access_and_evaluation.txt` — source permissions and retrieval/answer evaluation
- `pdf_ingestion_note.pdf` — a generated one-page PDF that exercises the PDF loader

The PDF can be regenerated with:

```bash
python scripts/generate_sample_pdf.py
```

Point the service at another directory of supported files with `RAG_DATA_DIR`:

```bash
RAG_DATA_DIR=/path/to/documents uvicorn multimodal_rag.api:app --port 8000
```

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATA_DIR` | `data/sample_docs` | Directory loaded at startup |
| `RAG_EMBEDDER` | `auto` | `auto`, `sentence-transformer`, or `hashing` |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RAG_RERANKER` | `lexical` | `lexical`, `cross-encoder`, or `none` |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural reranker model |
| `RAG_CHUNK_SIZE` | `900` | Maximum characters per chunk |
| `RAG_CHUNK_OVERLAP` | `140` | Character overlap target between chunks |
| `RAG_CANDIDATE_K` | `24` | Hybrid-search candidate pool size |
| `RAG_TOP_K` | `5` | Default final result count |

See `.env.example` for a copyable starting point.

## Project structure

```text
multimodal-rag-platform/
├── data/sample_docs/          # Synthetic PDF, Markdown, and text corpus
├── docs/architecture.md       # Design and production-extension notes
├── scripts/
│   ├── demo.py                # Source-checkout demo entry point
│   └── generate_sample_pdf.py # Regenerates the bundled sample PDF
├── src/multimodal_rag/
│   ├── agent.py               # Query planning and LangChain-compatible workflow
│   ├── api.py                 # FastAPI application
│   ├── chunking.py            # Paragraph-aware chunking
│   ├── cli.py                 # Command-line demo
│   ├── config.py              # Environment-driven settings
│   ├── embeddings.py          # Sentence-transformer and hashing embedders
│   ├── generation.py          # Citation-constrained extractive generator
│   ├── index.py               # FAISS/NumPy dense search, BM25, and RRF
│   ├── ingestion.py           # PDF, Markdown, and text loaders
│   ├── models.py              # Document, Chunk, SearchResult, Answer records
│   ├── reranking.py           # Cross-encoder and lexical rerankers
│   ├── service.py             # Shared application assembly
│   └── text.py                # Tokenization and text helpers
├── tests/
│   ├── test_api.py            # FastAPI health and ask-endpoint tests
│   └── test_pipeline.py       # Ingestion, chunking, retrieval, and grounding tests
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## API overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status, corpus size, active providers |
| `GET` | `/stats` | Document/chunk counts and index backend |
| `POST` | `/search` | Hybrid search without answer generation |
| `POST` | `/ask` | Grounded answer with citations and evidence scores |
| `POST` | `/documents` | Add one inline text document to the in-memory index |

The local `/documents` endpoint is for demos and tests. A production deployment should put ingestion behind authentication, authorization, validation, and an asynchronous indexing pipeline.

## Tests

Install development dependencies and run the suite:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The core tests use the hashing embedder and local BM25 fallback, so they do not need model downloads or cloud credentials. The API test runs when FastAPI and HTTPX are installed.

## Design choices worth noting

- **Hybrid retrieval is not a buzzword here.** Dense and BM25 rankings are produced independently and fused by rank, which avoids pretending cosine similarity and BM25 scores share a scale.
- **Grounding is testable.** The local generator is extractive, so an answer sentence can be traced to an exact chunk. A production LLM generator should be held to the same citation contract and evaluated for faithfulness.
- **Fallbacks are labeled.** FAISS falls back to exact NumPy search for tiny corpora and tests; sentence-transformers falls back to hashing embeddings only in `auto` mode when the transformer stack is unavailable. The active provider is exposed by `/health`.
- **The workflow is inspectable.** Planning, retrieval, fusion, reranking, and generation are separate stages rather than a single opaque prompt call.
- **No secrets are required.** The entire local path runs without AWS keys or a hosted LLM.

## Limitations

- The in-memory index is designed for the sample corpus and demonstrations, not large-scale serving.
- The local answer generator is extractive; it does not synthesize across passages as fluently as an LLM.
- PDF support extracts the embedded text layer. Scanned PDFs need an OCR extractor.
- Source-permission filtering, document deletion/versioning, evaluation dashboards, and persistent indexes are production concerns described in the architecture notes but not implemented in this compact reference version.

## License

MIT — see [`LICENSE`](LICENSE).
