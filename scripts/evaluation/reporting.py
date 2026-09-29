"""Formatting and text report generation for retrieval evaluations (EvalQueryV2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .matching import _is_legal_source

def _format_location(meta: dict[str, Any]) -> str:
    """Return a compact location string omitting NULL/empty fields."""
    parts = []
    if meta.get("chapter"):
        parts.append(str(meta["chapter"]))
    if meta.get("article"):
        parts.append(str(meta["article"]))
    if meta.get("clause"):
        parts.append(str(meta["clause"]))
    if meta.get("point"):
        parts.append(str(meta["point"]))
    return " / ".join(parts) if parts else ""


def _truncate(text: str, max_chars: int = 200) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "…"




def render_full_report_v2(
    *,
    query_records: list[Any],
    metrics: dict[str, Any],
    # Model / retrieval metadata
    model_name: str = "",
    model_alias: str = "",
    embedding_dim: int | None = None,
    backend: str = "",
    embedding_column: str = "",
    retriever: str = "HNSW (pgvector)",
    # Run metadata
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    top_k: int | None = None,
    ef_search: int | None = None,
    cli_args_str: str = "",
    # Database state
    state: dict[str, Any] | None = None,
    database_name: str = "",
    uncaught_exception: str | None = None,
) -> str:
    """Render a standardized, auditable report for EvalQueryV2 evaluations.

    Includes full model metadata, dataset summary, global metrics,
    per-category, per-difficulty, OOS/invalid, errors, and per-query details.
    """
    from .metrics import _get_chunk_id

    _state = state or {}
    total_chunks = _state.get("total_chunks", 0)
    embedded_chunks = _state.get("embedded_chunks", 0)
    missing_embeddings = _state.get("missing_embeddings", 0)
    hnsw_exists = _state.get("hnsw_index_exists", False)

    lines: list[str] = []

    # ==========================================================
    # Header
    # ==========================================================
    lines.append("=" * 60)
    lines.append("RETRIEVAL EVALUATION RESULT")
    lines.append("=" * 60)
    lines.append("")

    if start_time:
        lines.append(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if end_time:
        lines.append(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()
        lines.append(f"Duration:   {duration:.2f}s")
    lines.append("")

    # ==========================================================
    # Model Information
    # ==========================================================
    lines.append("Model Information")
    lines.append("-" * 17)
    lines.append(f"Model:             {model_name or 'N/A'}")
    lines.append(f"Model Alias:       {model_alias or 'N/A'}")
    lines.append(f"Embedding Dimension: {embedding_dim if embedding_dim is not None else 'N/A'}")
    lines.append(f"Backend:           {backend or 'N/A'}")
    lines.append(f"Embedding Column:  {embedding_column or 'N/A'}")
    lines.append(f"Retriever:         {retriever or 'N/A'}")
    if database_name:
        lines.append(f"Database:          {database_name}")
    if top_k is not None:
        lines.append(f"Top-K:             {top_k}")
    if ef_search is not None:
        lines.append(f"ef_search:         {ef_search}")
    if cli_args_str:
        lines.append(f"CLI arguments:     {cli_args_str}")
    lines.append("")

    if total_chunks > 0 or embedded_chunks > 0:
        coverage_pct = (embedded_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
        lines.append("Database State")
        lines.append("-" * 14)
        lines.append(f"Total chunks:      {total_chunks}")
        lines.append(f"Embedded chunks:   {embedded_chunks}")
        lines.append(f"Missing:           {missing_embeddings}")
        lines.append(f"Coverage:          {coverage_pct:.1f}% ({embedded_chunks}/{total_chunks})")
        lines.append(f"HNSW index:        {'EXISTS' if hnsw_exists else 'NOT FOUND'}")
        lines.append("")

    # ==========================================================
    # Evaluation Dataset
    # ==========================================================
    total_q = metrics.get("total_queries", len(query_records))
    answerable_q = metrics.get("evaluated_queries", 0)
    oos_q = metrics.get("unanswerable_queries", 0)
    min_grade = metrics.get("min_grade", 2)

    lines.append("Evaluation Dataset")
    lines.append("-" * 18)
    lines.append("Dataset:                    EvalQueryV2")
    lines.append(f"Total Queries:              {total_q}")
    lines.append(f"Answerable Queries:         {answerable_q}")
    lines.append(f"OOS / Invalid Queries:      {oos_q}")
    lines.append(f"Minimum Relevance Grade:    {min_grade}")
    lines.append("  (grade >= 2 -> relevant for Hit/Recall/Precision/MRR)")
    lines.append("  (grades 1/2/3 used for nDCG)")
    lines.append("")

    # ==========================================================
    # Global Metrics
    # ==========================================================
    METRIC_NAMES = (
        "Hit@3", "Hit@5", "Hit@10",
        "Recall@3", "Recall@5", "Recall@10",
        "Precision@3", "Precision@5", "Precision@10",
        "MRR",
        "nDCG@3", "nDCG@5", "nDCG@10",
    )

    lines.append("Global Metrics")
    lines.append("-" * 14)
    if answerable_q > 0:
        for name in METRIC_NAMES:
            value = metrics.get(name)
            if value is not None:
                lines.append(f"{name}: {value:.4f}")
            else:
                lines.append(f"{name}: N/A")
        top1_avg = metrics.get("Average_top1_similarity")
        if top1_avg is not None:
            lines.append(f"Average Top-1 Similarity: {top1_avg:.6f}")
    else:
        lines.append("N/A (no answerable queries evaluated)")
    lines.append("")

    # ==========================================================
    # Per-Category Results
    # ==========================================================
    lines.append("Per-Category Results")
    lines.append("-" * 20)
    per_category = metrics.get("per_category", {})
    if per_category:
        for category, values in sorted(per_category.items()):
            lines.append(f"  {category} (n={values.get('count', 0)})")
            for name in METRIC_NAMES:
                value = values.get(name)
                if value is not None:
                    lines.append(f"    {name}: {value:.4f}")
                else:
                    lines.append(f"    {name}: N/A")
            lines.append("")
    else:
        lines.append("  (no per-category data)")
        lines.append("")

    # ==========================================================
    # Per-Difficulty Results
    # ==========================================================
    lines.append("Per-Difficulty Results")
    lines.append("-" * 22)
    per_difficulty = metrics.get("per_difficulty", {})
    if per_difficulty:
        for difficulty in ("easy", "medium", "hard"):
            if difficulty not in per_difficulty:
                continue
            values = per_difficulty[difficulty]
            lines.append(f"  {difficulty} (n={values.get('count', 0)})")
            for name in METRIC_NAMES:
                value = values.get(name)
                if value is not None:
                    lines.append(f"    {name}: {value:.4f}")
                else:
                    lines.append(f"    {name}: N/A")
            lines.append("")
    else:
        lines.append("  (no per-difficulty data)")
        lines.append("")

    # ==========================================================
    # OOS / Invalid Queries
    # ==========================================================
    lines.append("OOS / Invalid Queries")
    lines.append("-" * 21)
    oos_invalid = metrics.get("oos_invalid", {})
    if not oos_invalid:
        lines.append("  No out-of-scope or invalid queries.")
    else:
        for category, values in sorted(oos_invalid.items()):
            count = values.get("count", 0)
            with_results = values.get("queries_with_results", 0)
            errors = values.get("retrieval_errors", 0)
            lines.append(
                f"  {category}: {count} queries, "
                f"{with_results} returned results, "
                f"{errors} retrieval errors"
            )
            for rec in values.get("records", []):
                q = rec.query
                top1_chunk = None
                if rec.results:
                    top1_chunk = _get_chunk_id(rec.results[0])
                err_str = f" [ERROR: {rec.error}]" if rec.error else ""
                lines.append(
                    f"    [{q.category}/{q.difficulty}] {q.query[:80]} "
                    f"-> top-1: {top1_chunk or 'N/A'}{err_str}"
                )
    lines.append("")

    # ==========================================================
    # Errors
    # ==========================================================
    errors = metrics.get("errors", [])
    lines.append("Errors")
    lines.append("-" * 6)
    if errors:
        for q_idx, q_txt, q_err in errors:
            lines.append(f"  Query {q_idx:03d} ({q_txt[:50]}...): {q_err}")
    else:
        lines.append("  None")
    lines.append("")

    if uncaught_exception:
        lines.append("=" * 60)
        lines.append("UNCAUGHT EXCEPTION")
        lines.append("=" * 60)
        lines.append("")
        lines.append(uncaught_exception)
        lines.append("")

    # ==========================================================
    # Per-Query Detail
    # ==========================================================
    lines.append("=" * 60)
    lines.append("PER-QUERY DETAIL")
    lines.append("=" * 60)
    lines.append("")

    FIELD_MAP = {
        "Hit@3": "hit3", "Hit@5": "hit5", "Hit@10": "hit10",
        "Recall@3": "recall3", "Recall@5": "recall5", "Recall@10": "recall10",
        "Precision@3": "precision3", "Precision@5": "precision5", "Precision@10": "precision10",
        "MRR": "mrr", "nDCG@3": "ndcg3", "nDCG@5": "ndcg5", "nDCG@10": "ndcg10",
    }

    for rec in query_records:
        query = rec.query
        lines.append("#" * 60)
        lines.append(f"QUERY {rec.index:03d} / {rec.total:03d}")
        lines.append("#" * 60)
        lines.append("")
        lines.append(f"Category:   {query.category}")
        lines.append(f"Difficulty: {query.difficulty}")
        lines.append(f"Answerable: {'yes' if query.is_answerable else 'no'}")
        lines.append(f"Question:   {query.query}")
        if query.description:
            lines.append(f"Info need:  {query.description}")
        if rec.error:
            lines.append(f"Retrieval error: {rec.error}")

        if query.relevant_chunks:
            lines.append("Ground-truth chunks (chunk_id: grade):")
            for chunk_id, grade in sorted(query.relevant_chunks.items()):
                lines.append(f"  {chunk_id}: {grade}")
        else:
            lines.append("Ground truth: none (unanswerable query)")

        lines.append("")
        lines.append("-" * 60)
        lines.append("RETRIEVAL RESULTS")
        lines.append("-" * 60)
        lines.append("")

        if not rec.results:
            lines.append("  (no results returned)")
        else:
            for rank, result in enumerate(rec.results, start=1):
                chunk_id = _get_chunk_id(result) or "N/A"
                grade = query.relevant_chunks.get(chunk_id, 0) if query.relevant_chunks else 0
                score = getattr(result, "score", None)
                score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
                meta = getattr(result, "metadata", None) or {}

                lines.append(f"Rank {rank}")
                lines.append(f"Score:    {score_text}")
                lines.append(f"Chunk ID: {chunk_id}")
                lines.append(f"Grade:    {grade}")

                if _is_legal_source(meta):
                    if meta.get("document_title"):
                        lines.append(f"Document: {meta['document_title']}")
                    if meta.get("article"):
                        lines.append(f"Article:  {meta['article']}")
                    if meta.get("clause"):
                        lines.append(f"Clause:   {meta['clause']}")
                    if meta.get("point"):
                        lines.append(f"Point:    {meta['point']}")
                else:
                    lines.append("Source:   QA")

                lines.append("")
                lines.append("Text:")
                text = getattr(result, "text", "")
                lines.append(text)
                lines.append("")
                lines.append("-" * 60)
                lines.append("")

        # Per-query summary
        lines.append("-" * 60)
        lines.append("QUERY SUMMARY")
        lines.append("-" * 60)
        lines.append("")
        lines.append(f"Best similarity:      {f'{rec.best_score:.6f}' if rec.best_score is not None else 'N/A'}")
        if query.is_answerable:
            frr = rec.first_relevant_rank
            lines.append(f"First relevant rank:  {frr if frr is not None else 'NOT FOUND'}")
            for name, field_name in FIELD_MAP.items():
                value = getattr(rec, field_name, None)
                if value is not None:
                    lines.append(f"{name}: {value:.4f}")
        else:
            lines.append("Metrics:              N/A (unanswerable query)")
        lines.append("")

    lines.append("="* 60)
    lines.append("END OF RESULT")
    lines.append("="* 60)
    lines.append("")
    return "\n".join(lines)
