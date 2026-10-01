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
else:
    from environment import load_dotenv as _load_env_file, require_database_url
    from evaluation.eval_queries_v2 import EVAL_QUERIES_V2
    from evaluation.runner import QueryEvalRecordV2, run_evaluation_v2
    from evaluation.reporting import render_full_report_v2
    from evaluation.matching import _is_legal_source
    from evaluation.metrics import _get_chunk_id

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
        default=10,
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
        choices=["hnsw", "bm25"],
        help="Retrieval method to use (default: hnsw).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Ad-hoc single-query mode
# ---------------------------------------------------------------------------


def _run_adhoc_query(retriever: Any, query: str, top_k: int) -> int:
    """Retrieve top-K results for *query* and print them; no metrics, no file."""
    print()
    print("=" * 60)
    print("AD-HOC QUERY")
    print("=" * 60)
    print(f"Query:  {query}")
    print(f"Top-K:  {top_k}")
    print()

    try:
        results = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:
        print(f"[ERROR] Retrieval failed: {exc}")
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
            if args.method == "bm25":
                if __package__:
                    from .retrievers.bm25 import BM25Retriever as Retriever  # noqa: PLC0415
                else:
                    from retrievers.bm25 import BM25Retriever as Retriever  # noqa: PLC0415
            else:
                if __package__:
                    from .retrievers.hnsw import Retriever  # noqa: PLC0415
                else:
                    from retrievers.hnsw import Retriever  # noqa: PLC0415
        except ImportError as exc:
            print(f"ERROR: Could not import retrieval module: {exc}")
            return 1

        if args.method == "bm25":
            retriever = Retriever(database_url=database_url)
        else:
            retriever = Retriever(
                database_url=database_url, ef_search=args.ef_search
            )
        return _run_adhoc_query(retriever, args.query, args.top_k)


    try:
        database_url = _get_database_url()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    database_name = _get_database_name(database_url)

    try:
        if args.method == "bm25":
            if __package__:
                from .retrievers.bm25 import BM25Retriever as Retriever  # noqa: PLC0415
            else:
                from retrievers.bm25 import BM25Retriever as Retriever  # noqa: PLC0415
        else:
            if __package__:
                from .retrievers.hnsw import Retriever  # noqa: PLC0415
            else:
                from retrievers.hnsw import Retriever  # noqa: PLC0415
    except ImportError as exc:
        print(f"ERROR: Could not import retrieval module: {exc}")
        return 1

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION (EvalQueryV2)")
    print("=" * 60)
    print(f"Method:     {args.method.upper()}")
    print(f"Top-K:      {args.top_k}")
    if args.method == "hnsw":
        print(f"ef_search:  {args.ef_search}")
    print(f"min_grade:  {args.min_grade}")
    print(f"Dataset:    EvalQueryV2 ({len(EVAL_QUERIES_V2)} queries)")
    print("INFO: Initialising retriever...")

    if args.method == "bm25":
        retriever = Retriever(database_url=database_url)
    else:
        retriever = Retriever(database_url=database_url, ef_search=args.ef_search)
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
    if hasattr(retriever, "verify_database_state"):
        state = retriever.verify_database_state()
        print(f"Total chunks:       {state['total_chunks']}")
        print(f"Embedded chunks:    {state['embedded_chunks']}")
        print(f"Missing embeddings: {state['missing_embeddings']}")
        print(f"HNSW index exists:  {state['hnsw_index_exists']}")
    else:
        state = {"total_chunks": getattr(retriever, "corpus_size", "N/A")}
        print(f"Total chunks:       {state['total_chunks']}")

    queries = EVAL_QUERIES_V2
    print(f"\nRunning {len(queries)} evaluation queries...\n")

    query_records: list[QueryEvalRecordV2] = []
    metrics: dict[str, Any] = {}
    uncaught_exception: str | None = None

    try:
        metrics = run_evaluation_v2(
            queries, retriever, top_k=args.top_k, min_grade=args.min_grade
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
            retriever="BM25 (rank_bm25)" if args.method == "bm25" else "HNSW (pgvector)",
            start_time=start_time,
            end_time=end_time,
            top_k=args.top_k,
            ef_search=args.ef_search,
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
    if metrics.get("Average_top1_similarity") is not None:
        print(f"Avg top-1 sim: {metrics['Average_top1_similarity']:.6f}")

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print(f"Log saved to: logs/{log_path.name}")
    print("=" * 60)

    if uncaught_exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
