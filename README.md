# RAG-based Q&A System for Decree 116/2020/NĐ-CP

## Introduction

This project builds a legal Q&A system focused on **Decree 116/2020/NĐ-CP** on support policies for pedagogy students. The dataset includes the main decree, **Decree 60/2025/NĐ-CP** (amending document), and the **Education Law 2019** to supplement definitions and legal context. The Q&A set on Decree 116 is used as reference material, not a substitute for official legal sources.

Following the RAG (Retrieval-Augmented Generation) approach, the system receives a question, retrieves relevant legal passages, and generates a Vietnamese answer with source citations via an **LLM**.

---

## Overall Architecture — Complete Pipeline

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE / BUILD PHASE                            │
│                                                                          │
│  Legal Markdown files (data/markdown/, data/raw/qa/)                     │
│          ↓                                                               │
│  scripts/legal_chunker.py  →  data/processed/legal_chunks.jsonl          │
│          ↓                                                               │
│  scripts/validate_legal_chunks.py  →  validate JSONL (618 chunks)        │
│          ↓                                                               │
│  scripts/import_legal_data.py  →  PostgreSQL (documents, legal_chunks)   │
│          ↓                                                               │
│  scripts/index_embeddings.py  →  BGE-M3 embeddings + HNSW index          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         ONLINE / QUERY PHASE                             │
│                                                                          │
│  User question in Vietnamese                                             │
│          ↓                                                               │
│  RAGPipeline.run() — scripts/rag_pipeline.py                             │
│          ↓                                                               │
│  HybridRetriever (HNSW + BM25 + RRF)  →  top-40 candidates               │
│          ↓                                                               │
│  CrossEncoderReranker (bge-reranker-v2-m3)  →  top-10 chunks             │
│          ↓                                                               │
│  format_evidence_blocks()  →  numbered evidence [1]…[n] + metadata       │
│          ↓                                                               │
│  build_user_prompt()  →  LEGAL SOURCES + QUESTION                        │
│          ↓                                                               │
│  LLMClient.generate()  →  Vietnamese answer                              │
│          ↓                                                               │
│  RAGResponse(answer, sources, retrieval_method, retrieved_count)         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Technologies

| Component | Technology |
|---|---|
| Database | PostgreSQL + pgvector |
| Embedding model | `BAAI/bge-m3` (default) and 4 alternative models |
| Vector index | HNSW (`vector_cosine_ops`, m=16, ef_construction=64) |
| Keyword search | BM25 (`pg_trgm` + PostgreSQL full-text) |
| Fusion & Reranking | Hybrid RRF + Cross-Encoder `BAAI/bge-reranker-v2-m3` |
| Generation model | LLM (configured via `.env`) |
| Language | Python |
| DB connection | `psycopg` / `psycopg2` |

---

## Running the System

### Step 0 — Activate the environment (optional)

Example using Conda:

```powershell
conda activate chatbot
```

### Steps 1–4: Build Phase (Offline)

**1. Chunk legal data:**

```powershell
python scripts/legal_chunker.py
```

**2. Validate JSONL:**

```powershell
python scripts/validate_legal_chunks.py
```

**3. Import into PostgreSQL:**

```powershell
python scripts/import_legal_data.py
```

> Requires PostgreSQL to be running and `DATABASE_URL` configured in `.env`.

**4. Generate embeddings & build HNSW index:**

```powershell
python scripts/index_embeddings.py --model bge-m3
```

### Step 5: Evaluate Retrieval (optional)

```powershell
# HNSW only
$env:EMBEDDING_MODEL = "bge-m3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# Hybrid RRF
python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80

# Hybrid + Cross-Encoder Rerank (best benchmark configuration)
python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

Results are written to `logs/`.

### Step 6: Run the End-to-End RAG Pipeline

Using a Python script:

```python
from scripts.rag_pipeline import build_pipeline, run_query

# Initialize the pipeline (load embedding model + BM25 corpus — ~30-60s on first run)
pipeline = build_pipeline()

# Query
response = run_query(pipeline, "How much support do pedagogy students receive?")

print(response.answer)
```

Or directly from the command line:

```powershell
python -c "
from scripts.rag_pipeline import build_pipeline, run_query
p = build_pipeline()
r = run_query(p, 'How much support do pedagogy students receive?')
print(r.answer)
"
```

> Requires LLM API key and model to be configured in `.env` (see `.env.example`).

---

## Default Pipeline Configuration (`scripts/config.py`)

| Parameter | Value | Description |
|---|---:|---|
| `CANDIDATE_K` | 40 | Candidates per retriever (HNSW, BM25) before RRF |
| `TOP_K` | 20 | Results after RRF, before reranking |
| `RRF_K` | 60 | RRF constant |
| `EF_SEARCH` | 80 | HNSW ef_search |
| `RERANK` | True | Enable Cross-Encoder reranking by default |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-Encoder model |
| `RERANK_TOP_K` | 10 | Number of chunks passed to the Generation layer |

---

## Retrieval Method — Metadata-aware Embedding

Each legal passage is represented with a structural prefix before being fed into BGE-M3:

```text
[Document] <document_title>
[Chapter]  <chapter>
[Article]  <article>
[Clause]   <clause>
[Point]    <point>
[Content]  <original content>
```

Questions are embedded verbatim, without transformation. Details: [Retrieval Development History](docs/retrieval-development-history.md).

---

## Retrieval Evaluation Results — EvalQueryV2

### Best Benchmark — Hybrid RRF + Cross-Encoder Rerank (2026-10-06)

Configuration: `candidate-k=40`, `top-k=20`, `rrf-k=60`, `ef_search=80`, reranker `BAAI/bge-reranker-v2-m3`, `rerank-top-k=10`.

| Model | Alias | Hit@3 | Hit@10 | Recall@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | `bge-m3` | **0.8421** | 0.9053 | **0.6169** | **0.7568** | **0.5713** |
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 0.8316 | **0.9368** | 0.6124 | 0.7299 | 0.5626 |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | **0.8421** | 0.9158 | 0.5944 | 0.7362 | 0.5468 |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 0.7895 | 0.8632 | 0.5385 | 0.7028 | 0.5318 |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 0.7789 | 0.8526 | 0.5176 | 0.6875 | 0.5023 |

**Observations:**
- `bge-m3` is the strongest overall: Recall@10 (0.6169), MRR (0.7568), nDCG@10 (0.5713) → **default model**.
- `vietlegal-harrier` has the highest Hit@10 (0.9368), suitable when coverage is the priority.
- Full details: [Retrieval Evaluation](docs/retrieval-evaluation.md).

**Reproduction command:**
```powershell
$env:EMBEDDING_MODEL="bge-m3"; python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

---

## Generation Layer — Grounded Answers with LLM

`scripts/rag_pipeline.py` orchestrates the full pipeline from question to answer:

1. **Validate** the input question.
2. **Retrieve** relevant chunks via HybridRetriever (HNSW + BM25 + RRF).
3. **Rerank** with Cross-Encoder `bge-reranker-v2-m3` → top-10 chunks.
4. **Format** chunks into numbered evidence blocks `[S1]…[Sn]` with legal metadata.
5. **Build prompt** combining `LEGAL SOURCES` + question.
6. **Generate** a Vietnamese answer with inline citations `[1], [2]…` via LLM client (default `GeminiClient`).
7. **Return** `RAGResponse(answer, sources, retrieval_method, retrieved_count)`.

### Required Environment Variables

| Variable | Description |
|---|---|
| `LLM_API_KEY` | API key of the LLM provider (e.g. Gemini, OpenRouter…) |
| `LLM_MODEL` | LLM model name (e.g. `gemini-2.0-flash`) |
| `DATABASE_URL` | PostgreSQL connection URL |
| `EMBEDDING_MODEL` | Embedding model alias (default `bge-m3`) |

> Exact variable names depend on the LLM client in use. See `.env.example` for details.

### Citations and Grounding

- Each chunk is labeled `[1]`, `[2]`… in the prompt. The LLM is instructed to cite using these exact labels and append a **References** section at the end.
- Legal metadata (article, clause, point, document name) is displayed in the header of each evidence block.
- The system **does not infer** clauses when metadata is missing — a deliberate design choice to prevent hallucination.

Details: [Generation Layer](docs/generation.md).

---

## Directory Structure

```text
init.sql                     ← PostgreSQL schema (source of truth)
.env.example                 ← environment variable configuration template
requirements.txt
README.md
docs/                        ← detailed documentation
scripts/
├── config.py                ← centralized pipeline configuration
├── environment.py           ← load environment variables (.env)
├── database.py              ← PostgreSQL connection
├── retrieval_types.py       ← shared RetrievalResult contract
├── rag_pipeline.py          ← end-to-end RAG pipeline (Retrieval → Generation)
├── data_pipeline/           ← chunking, validation
├── embeddings/              ← embedding backend, model registry
├── evaluation/              ← dataset, matching, metrics, runner, reporting
├── generator/
│   ├── gemini.py            ← Gemini API wrapper
│   └── openrouter.py        ← OpenRouter API wrapper (fallback)
├── indexing/                ← embedding_index, import_data
├── retrievers/              ← bm25, hnsw, hybrid, reranker
├── legal_chunker.py         ← chunking CLI wrapper
├── import_legal_data.py     ← import_data CLI wrapper
├── validate_legal_chunks.py ← validation CLI wrapper
├── index_embeddings.py      ← indexing CLI wrapper
└── test_retrieval.py        ← HNSW, BM25, Hybrid evaluation CLI (EvalQueryV2)
tests/
├── test_embedding.py        ← embedding unit tests
├── test_retrieval.py        ← HNSW retrieval unit tests
├── test_bm25_retriever.py   ← BM25 unit tests
├── test_hybrid_retriever.py ← RRF, Hybrid, evaluation integration unit tests
└── test_generation.py       ← Generation unit tests (Gemini mock)
logs/
├── retrieval_*.txt          ← timestamped evaluation reports
└── <model>_<timestamp>.txt  ← benchmark report per embedding model
data/
├── markdown/                ← original legal documents (.md)
├── raw/qa/                  ← reference Q&A set
└── processed/
    └── legal_chunks.jsonl   ← 618 processed chunks
```

---

## Detailed Documentation

| Document | Contents |
|---|---|
| [Setup & Configuration](docs/setup.md) | Environment requirements, Conda, pip, `.env` variables |
| [Data Preparation](docs/data-preparation.md) | Legal sources, chunking pipeline, JSONL |
| [Database & Import](docs/database-and-import.md) | `init.sql` schema, import, DB verification |
| [Embedding & Indexing](docs/embedding-and-indexing.md) | BGE-M3, HNSW, configuration, indexing results |
| [Embedding Model Switching](docs/embedding-model-switching.md) | Guide to switching between embedding models |
| [Retrieval](docs/retrieval.md) | Architecture, modules, usage, unit tests |
| [BM25 Retrieval](docs/retrieval-bm25.md) | BM25 details, configuration, comparison with HNSW |
| [Hybrid Search](docs/retrieval-hybrid.md) | HNSW + BM25, RRF, configuration and diagnostics |
| [Retrieval Development History](docs/retrieval-development-history.md) | Text-only → metadata-aware → hybrid |
| [Retrieval Evaluation](docs/retrieval-evaluation.md) | Methodology, metrics, 5-model benchmark, in-depth analysis |
| [Generation Layer](docs/generation.md) | Gemini wrapper, RAG pipeline, grounding, citations |
