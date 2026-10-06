"""Live retrieval evaluation script for the legal RAG chatbot (EvalQueryV2).

Runs the full EvalQueryV2 benchmark (100 queries) through the Retriever and
produces a detailed ``.txt`` report under ``logs/``.

Run from the repository root::

    conda activate chatbot
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --top-k 10 --ef-search 80

Dataset
-------
The evaluation dataset is ``EvalQueryV2`` defined in
``scripts/evaluation/eval_queries_v2.py``.  It contains 100 queries spanning
six categories: exact, semantic, contextual, multi_chunk, multi_document,
complex_qa, plus out_of_scope and invalid queries.

Ground truth
------------
Each answerable query specifies a ``relevant_chunks`` dict of
``{chunk_id: grade}`` with grades 1-3:
  3 = directly answers the query
  2 = relevant / supporting information
  1 = weakly relevant / context

Metrics
-------
Hit@K, Recall@K, Precision@K, and MRR use grade >= 2 as the relevance
threshold (``min_grade=2``).  nDCG@K uses the full graded relevance.
OOS and invalid queries are excluded from all metric aggregates.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from secrets import randbelow

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent

if __package__:
    from .environment import load_dotenv as _load_env_file, require_database_url
    from .evaluation.eval_queries_v2 import EVAL_QUERIES_V2
    from .evaluation.runner import QueryEvalRecordV2, run_evaluation_v2
    from .evaluation.reporting import render_full_report_v2
    from .evaluation.matching import _is_legal_source
    from .evaluation.metrics import _get_chunk_id
    from .retrievers.hybrid import (
        DEFAULT_CANDIDATE_K,
        DEFAULT_FINAL_TOP_K,
        DEFAULT_RRF_K,
    )
    from .retrievers.reranker import (
        CrossEncoderReranker,
        DEFAULT_RERANK_MODEL,
        DEFAULT_RERANK_TOP_K,
    )
else:
    from environment import load_dotenv as _load_env_file, require_database_url
    from evaluation.eval_queries_v2 import EVAL_QUERIES_V2
    from evaluation.runner import QueryEvalRecordV2, run_evaluation_v2
    from evaluation.reporting import render_full_report_v2
    from evaluation.matching import _is_legal_source
    from evaluation.metrics import _get_chunk_id
    from retrievers.hybrid import (
        DEFAULT_CANDIDATE_K,
        DEFAULT_FINAL_TOP_K,
        DEFAULT_RRF_K,
    )
    from retrievers.reranker import (
        CrossEncoderReranker,
        DEFAULT_RERANK_MODEL,
        DEFAULT_RERANK_TOP_K,
    )

LOGGER = logging.getLogger("test_retrieval")


# Helpers
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    _load_env_file(path)


def _get_database_url() -> str:
    return require_database_url(
        ROOT / ".env",
        loader=_load_dotenv,
        error_message="DATABASE_URL is required.  Set it in the environment or in .env",
    )


def _get_database_name(url: str) -> str:
    try:
        parsed = urlparse(url)
        name = parsed.path.lstrip("/")
        return name if name else "PostgreSQL"
    except Exception:
        return "PostgreSQL"


def _get_model_metadata(retriever: Any) -> dict[str, Any]:
    """Extract model metadata for report headers."""
    em = getattr(retriever, "_embedding_model", None)
    cfg = getattr(em, "config", None)

    model_name = getattr(em, "model_name", "N/A") if em else "N/A"
    model_alias = getattr(cfg, "alias", "N/A") if cfg else "N/A"
    backend = getattr(cfg, "backend", "N/A") if cfg else "N/A"
    embedding_column = getattr(cfg, "embedding_column", "N/A") if cfg else "N/A"
    embedding_dim = getattr(em, "embedding_dim", 1024) if em else 1024

    return {
        "model_name": model_name,
        "model_alias": model_alias,
        "backend": backend,
        "embedding_column": embedding_column,
        "embedding_dim": embedding_dim,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieval evaluation against EvalQueryV2 for the legal RAG chatbot. "
            "Use --query for a quick single-query lookup without metrics or a log file."
        )
    )
    parser.add_argument(
        "--query",
        default=None,
        metavar="TEXT",
        help=(
            "Run a single ad-hoc query and print the top-K results to the terminal. "
            "No ground truth, no metrics, and no .txt file are generated."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_FINAL_TOP_K,
        metavar="N",
        help="Number of results to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=80,
        metavar="N",
        help="HNSW ef_search value (default: 80).  Higher = better recall, slower.",
    )
    parser.add_argument(
        "--min-grade",
        type=int,
        default=2,
        metavar="G",
        help="Minimum relevance grade for Hit/Recall/Precision/MRR (default: 2).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hnsw",
        choices=["hnsw", "bm25", "hybrid"],
        help="Retrieval method to use (default: hnsw).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        metavar="N",
        help=f"Candidates requested from each retriever in hybrid mode (default: {DEFAULT_CANDIDATE_K}).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        metavar="N",
        help=f"RRF rank constant in hybrid mode (default: {DEFAULT_RRF_K}).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help=(
            "Apply Cross-Encoder re-ranking after hybrid retrieval. "
            "Only valid with --method hybrid."
        ),
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=DEFAULT_RERANK_MODEL,
        metavar="MODEL",
        help=f"HuggingFace Cross-Encoder model for re-ranking (default: {DEFAULT_RERANK_MODEL}).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=DEFAULT_RERANK_TOP_K,
        metavar="N",
        help=f"Number of results to keep after re-ranking (default: {DEFAULT_RERANK_TOP_K}).",
    )
    return parser.parse_args(argv)


def _build_retriever(
    method: str,
    database_url: str,
    ef_search: int,
    candidate_k: int,
    rrf_k: int,
) -> tuple[Any, Any | None, Any | None]:
    """Build the selected retriever and return its dense/sparse components."""
    if __package__:
        from .retrievers.bm25 import BM25Retriever
        from .retrievers.hnsw import Retriever as HNSWRetriever
        from .retrievers.hybrid import HybridRetriever
    else:
        from retrievers.bm25 import BM25Retriever
        from retrievers.hnsw import Retriever as HNSWRetriever
        from retrievers.hybrid import HybridRetriever

    if method == "bm25":
        sparse = BM25Retriever(database_url=database_url)
        return sparse, None, sparse
    if method == "hybrid":
        dense = HNSWRetriever(database_url=database_url, ef_search=ef_search)
        sparse = BM25Retriever(database_url=database_url)
        hybrid = HybridRetriever(
            {"hnsw": dense, "bm25": sparse},
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )
        return hybrid, dense, sparse
    dense = HNSWRetriever(database_url=database_url, ef_search=ef_search)
    return dense, dense, None


# ---------------------------------------------------------------------------
# Ad-hoc single-query mode
# ---------------------------------------------------------------------------


def _run_adhoc_query(
    retriever: Any,
    query: str,
    top_k: int,
    *,
    reranker: Any | None = None,
) -> int:
    """Retrieve top-K results for *query* and print them; no metrics, no file.

    If *reranker* is provided (a ``CrossEncoderReranker`` instance), the
    retrieval candidates are re-scored and trimmed to ``reranker.rerank_top_k``
    before printing.
    """
    print()
    print("=" * 60)
    print("AD-HOC QUERY")
    print("=" * 60)
    print(f"Query:  {query}")
    print(f"Top-K:  {top_k}")
    if reranker is not None:
        print(f"Re-ranker: {reranker.model_name} (rerank_top_k={reranker.rerank_top_k})")
    print()

    try:
        results = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:
        print(f"[ERROR] Retrieval failed: {exc}")
        return 1

    # Apply re-ranking if requested.
    if reranker is not None and results:
        try:
            results = reranker.rerank(query, results)
        except Exception as exc:
            print(f"[ERROR] Re-ranking failed: {exc}")
            return 1

    if not results:
        print("(no results returned)")
        return 0

    for rank, result in enumerate(results, start=1):
        chunk_id = _get_chunk_id(result) or "N/A"
        score = getattr(result, "score", None)
        score_str = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
        meta = getattr(result, "metadata", None) or {}
        text = getattr(result, "text", "") or ""

        print("-" * 60)
        print(f"Rank {rank}")
        print(f"Score:    {score_str}")
        print(f"Chunk ID: {chunk_id}")
        fusion = meta.get("retrieval_diagnostics") or {}
        if fusion:
            # Cross-Encoder re-ranking diagnostics (if --rerank was used).
            pre_score = fusion.get("pre_rerank_score")
            pre_rank = fusion.get("pre_rerank_rank")
            reranker_model = fusion.get("reranker_model")
            if reranker_model:
                print(f"CE score:         {score_str}")
                if pre_score is not None:
                    print(f"Pre-rerank score: {pre_score:.6f} (rank {pre_rank})")
                print(f"Re-ranker:        {reranker_model}")
            # RRF fusion diagnostics.
            rrf_score = fusion.get("pre_rerank_score") if reranker_model else fusion.get("rrf_score", score_str)
            source_ranks = fusion.get("source_ranks", {})
            if source_ranks:
                print(f"RRF score: {fusion.get('rrf_score', rrf_score)}")
                print(
                    "Source ranks: " + ", ".join(
                        f"{name}={value if value is not None else 'not retrieved'}"
                        for name, value in source_ranks.items()
                    )
                )

        if _is_legal_source(meta):
            if meta.get("document_title"):
                print(f"Document: {meta['document_title']}")
            if meta.get("article"):
                print(f"Article:  {meta['article']}")
            if meta.get("clause"):
                print(f"Clause:   {meta['clause']}")
            if meta.get("point"):
                print(f"Point:    {meta['point']}")
        else:
            print("Source:   QA")

        print()
        print("Text:")
        print(text)
        print()

    print("=" * 60)
    print(f"Done.  {len(results)} result(s) returned.")
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Reconfigure stdout to UTF-8 so Vietnamese characters print correctly on
    # Windows terminals that default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    start_time = datetime.now()
    cli_args_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(default arguments)"

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Ad-hoc mode: single query, no metrics, no .txt file.
    # ------------------------------------------------------------------
    if args.query:
        try:
            database_url = _get_database_url()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

        try:
            retriever, _, _ = _build_retriever(
                args.method, database_url, args.ef_search,
                args.candidate_k, args.rrf_k,
            )
        except (ImportError, ValueError) as exc:
            print(f"ERROR: Could not initialise retriever: {exc}")
            return 1

        # Optionally wrap with re-ranker for ad-hoc mode.
        reranker: CrossEncoderReranker | None = None
        if args.rerank:
            if args.method != "hybrid":
                print("WARNING: --rerank is only meaningful with --method hybrid. Ignoring.")
            else:
                try:
                    reranker = CrossEncoderReranker(
                        model_name=args.rerank_model,
                        rerank_top_k=args.rerank_top_k,
                    )
                except (ImportError, ValueError) as exc:
                    print(f"ERROR: Could not initialise re-ranker: {exc}")
                    return 1

        return _run_adhoc_query(retriever, args.query, args.top_k, reranker=reranker)


    try:
        database_url = _get_database_url()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    database_name = _get_database_name(database_url)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION (EvalQueryV2)")
    print("=" * 60)
    print(f"Method:     {args.method.upper()}")
    print(f"Top-K:      {args.top_k}")
    if args.method in ("hnsw", "hybrid"):
        print(f"ef_search:  {args.ef_search}")
    if args.method == "hybrid":
        print(f"candidate_k: {args.candidate_k}")
        print(f"RRF k:       {args.rrf_k}")
        if args.rerank:
            print(f"Re-ranker:   {args.rerank_model}")
            print(f"Rerank top-k: {args.rerank_top_k}")
    print(f"min_grade:  {args.min_grade}")
    print(f"Dataset:    EvalQueryV2 ({len(EVAL_QUERIES_V2)} queries)")
    print("INFO: Initialising retriever...")

    try:
        retriever, dense_retriever, sparse_retriever = _build_retriever(
            args.method, database_url, args.ef_search,
            args.candidate_k, args.rrf_k,
        )
    except (ImportError, ValueError) as exc:
        print(f"ERROR: Could not initialise retriever: {exc}")
        return 1
    if args.method == "bm25":
        model_name = "BM25"
        model_alias = "Okapi BM25"
        embedding_dim = 0
        device = "CPU"
        model_meta = {
            "model_name": model_name,
            "model_alias": model_alias,
            "backend": "rank_bm25",
            "embedding_column": "N/A",
            "embedding_dim": embedding_dim,
        }
    elif args.method == "hybrid":
        dense_meta = _get_model_metadata(dense_retriever)
        _rerank_suffix = f" + {args.rerank_model} Re-rank" if args.rerank else ""
        model_name = f"Hybrid_{dense_meta['model_name']}"
        model_alias = f"Hybrid RRF ({dense_meta['model_alias']} + BM25){_rerank_suffix}"
        embedding_dim = dense_meta["embedding_dim"]
        device = dense_retriever.device
        model_meta = {
            **dense_meta,
            "model_name": model_name,
            "model_alias": model_alias,
            "backend": "HNSW (pgvector) + rank_bm25" + (_rerank_suffix or ""),
        }
    else:
        model_meta = _get_model_metadata(retriever)
        model_name = model_meta["model_name"]
        model_alias = model_meta["model_alias"]
        embedding_dim = model_meta["embedding_dim"]
        device = retriever.device

    # Build a filesystem-safe model name for the output file.
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    safe_model_name = "_".join(safe_model_name.split())

    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    random_suffix = f"{randbelow(10000):04d}"
    log_path = logs_dir / f"{safe_model_name}_{timestamp}_{random_suffix}.txt"
    while log_path.exists():
        random_suffix = f"{randbelow(10000):04d}"
        log_path = logs_dir / f"{safe_model_name}_{timestamp}_{random_suffix}.txt"

    print(f"Model:            {model_name}")
    print(f"Embedding device: {device}")

    print("\n--- Database State ---")
    state_retriever = dense_retriever or sparse_retriever or retriever
    if hasattr(state_retriever, "verify_database_state"):
        state = state_retriever.verify_database_state()
        print(f"Total chunks:       {state['total_chunks']}")
        print(f"Embedded chunks:    {state['embedded_chunks']}")
        print(f"Missing embeddings: {state['missing_embeddings']}")
        print(f"HNSW index exists:  {state['hnsw_index_exists']}")
    else:
        state = {"total_chunks": getattr(state_retriever, "corpus_size", "N/A")}
        print(f"Total chunks:       {state['total_chunks']}")

    queries = EVAL_QUERIES_V2
    print(f"\nRunning {len(queries)} evaluation queries...\n")

    # Build re-ranker if requested (hybrid only).
    reranker_for_eval: CrossEncoderReranker | None = None
    if args.rerank:
        if args.method != "hybrid":
            print("WARNING: --rerank is only meaningful with --method hybrid. Ignoring.")
        else:
            print("INFO: Initialising Cross-Encoder re-ranker ...")
            try:
                reranker_for_eval = CrossEncoderReranker(
                    model_name=args.rerank_model,
                    rerank_top_k=args.rerank_top_k,
                )
            except (ImportError, ValueError) as exc:
                print(f"ERROR: Could not initialise re-ranker: {exc}")
                return 1

    # If re-ranking is active, wrap the retriever so run_evaluation_v2 sees a
    # single object with a .retrieve(query, top_k) interface that first fetches
    # top_k candidates then re-ranks them down to rerank_top_k.
    eval_retriever = retriever
    eval_top_k = args.top_k
    if reranker_for_eval is not None:

        class _RerankedRetriever:
            """Thin wrapper: HybridRetriever + CrossEncoderReranker."""

            def __init__(self, base: Any, rr: CrossEncoderReranker) -> None:
                self._base = base
                self._rr = rr

            def retrieve(self, query: str, top_k: int) -> list[Any]:
                candidates = self._base.retrieve(query, top_k=top_k)
                return self._rr.rerank(query, candidates)

        eval_retriever = _RerankedRetriever(retriever, reranker_for_eval)
        # After re-ranking the result list has at most rerank_top_k items;
        # pass that as top_k so metrics are computed over the final list size.
        eval_top_k = args.rerank_top_k

    query_records: list[QueryEvalRecordV2] = []
    metrics: dict[str, Any] = {}
    uncaught_exception: str | None = None

    try:
        metrics = run_evaluation_v2(
            queries, eval_retriever, top_k=eval_top_k, min_grade=args.min_grade
        )
        query_records = metrics.get("query_records", [])
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            uncaught_exception = "Evaluation was interrupted by user (KeyboardInterrupt)."
        else:
            uncaught_exception = traceback.format_exc()
        print(f"\n[EXCEPTION] {exc}")
    finally:
        end_time = datetime.now()
        report = render_full_report_v2(
            query_records=query_records,
            metrics=metrics,
            model_name=model_name,
            model_alias=model_alias,
            embedding_dim=embedding_dim,
            backend=model_meta["backend"],
            embedding_column=model_meta["embedding_column"],
            retriever={
                "bm25": "BM25 (rank_bm25)",
                "hnsw": "HNSW (pgvector)",
                "hybrid": (
                    f"Hybrid RRF (HNSW pgvector + BM25 rank_bm25) + "
                    f"CrossEncoder ({args.rerank_model}) Re-rank"
                    if args.rerank
                    else "Hybrid RRF (HNSW pgvector + BM25 rank_bm25)"
                ),
            }[args.method],
            start_time=start_time,
            end_time=end_time,
            top_k=args.top_k,
            ef_search=args.ef_search if args.method in ("hnsw", "hybrid") else None,
            candidate_k=args.candidate_k if args.method == "hybrid" else None,
            rrf_k=args.rrf_k if args.method == "hybrid" else None,
            top1_score_label={
                "hnsw": "Average Top-1 cosine similarity (diagnostic only)",
                "bm25": "Average Top-1 BM25 score (diagnostic only)",
                "hybrid": "Average Top-1 RRF score (diagnostic only)",
            }[args.method],
            cli_args_str=cli_args_str,
            state=state if 'state' in dir() else None,
            database_name=database_name,
            uncaught_exception=uncaught_exception,
        )
        log_path.write_text(report, encoding="utf-8")

    # Terminal summary
    print("\n" + "=" * 60)
    print("AGGREGATE EVALUATION SUMMARY (EvalQueryV2)")
    print("=" * 60)
    print(f"Total queries:          {metrics.get('total_queries', len(query_records))}")
    print(f"Answerable evaluated:   {metrics.get('evaluated_queries', 0)}")
    print(f"OOS / Invalid:          {metrics.get('unanswerable_queries', 0)}")
    if metrics.get("evaluated_queries", 0) > 0:
        for metric in (
            "Hit@3", "Hit@5", "Hit@10",
            "Recall@3", "Recall@5", "Recall@10",
            "Precision@3", "Precision@5", "Precision@10",
            "MRR", "nDCG@3", "nDCG@5", "nDCG@10",
        ):
            value = metrics.get(metric)
            if value is not None:
                print(f"{metric:<14}: {value:.4f}")
    if metrics.get("Average_top1_score") is not None:
        print(
            f"Avg top-1 score (diagnostic only): "
            f"{metrics['Average_top1_score']:.6f}"
        )

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print(f"Log saved to: logs/{log_path.name}")
    print("=" * 60)

    if uncaught_exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
