"""Centralised retrieval-pipeline configuration for the legal RAG chatbot.

All tuneable retrieval parameters live here so they can be changed in one
place and take effect across the pipeline (``rag_pipeline.py``) and the
evaluation CLI (``test_retrieval.py``).

Values are set to the **best-performing benchmark configuration** from the
2026-10-06 evaluation (Hybrid RRF + Cross-Encoder Rerank):

    python scripts/test_retrieval.py \\
        --method hybrid \\
        --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 \\
        --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10

See ``docs/retrieval-evaluation.md`` for the full benchmark results.

Low-level module defaults (``retrievers/hybrid.py``, ``retrievers/hnsw.py``,
``retrievers/bm25.py``, ``retrievers/reranker.py``) are intentionally kept
separate — they reflect per-module library defaults and are not overridden
here.  This file governs only the **pipeline-level** configuration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------

# Candidates fetched from each retriever (HNSW and BM25) before RRF fusion.
# Higher values increase recall at the cost of latency.
CANDIDATE_K: int = 40

# Final results returned by HybridRetriever after RRF fusion.
# These results are optionally passed through the reranker.
TOP_K: int = 20

# RRF rank constant.  Standard value; higher values reduce score differences
# between adjacent ranks.
RRF_K: int = 60

# HNSW ef_search.  Higher values improve recall at the cost of query latency.
EF_SEARCH: int = 80

# ---------------------------------------------------------------------------
# Cross-Encoder reranking
# ---------------------------------------------------------------------------

# Apply reranking by default in the pipeline (True = best quality).
RERANK: bool = True

# HuggingFace model identifier for the Cross-Encoder reranker.
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

# Number of results kept after reranking.  Passed to the generator as context.
RERANK_TOP_K: int = 10

# ---------------------------------------------------------------------------
# Generator context window
# ---------------------------------------------------------------------------

# Chunks forwarded to Gemini.  When reranking is enabled this is typically
# equal to RERANK_TOP_K; when reranking is disabled it equals TOP_K.
GENERATOR_TOP_K: int = RERANK_TOP_K
