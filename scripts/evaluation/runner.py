"""Run queries and compute benchmark aggregates.

Evaluation pipeline based on EvalQueryV2 (graded chunk-level ground truth).

All metrics are computed over answerable queries only (is_answerable=True).
OOS / invalid queries are evaluated for retrieval behaviour but are excluded
from Hit/Recall/Precision/MRR/nDCG aggregates and reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .eval_queries_v2 import EvalQueryV2
from .metrics import (
    _get_chunk_id,
    compute_hit_at_k_v2,
    compute_mrr_v2,
    compute_ndcg_at_k,
    compute_precision_at_k_v2,
    compute_recall_at_k_v2,
)



@dataclass
class QueryEvalRecordV2:
    """Per-query evaluation record for EvalQueryV2."""

    index: int
    total: int
    query: EvalQueryV2
    results: list[Any]
    error: str | None = None
    best_score: float | None = None
    diagnostics: dict[str, Any] | None = None
    # Graded metrics (min_grade=2 by default, i.e. grade>=2 counts)
    hit3: float | None = None
    hit5: float | None = None
    hit10: float | None = None
    recall3: float | None = None
    recall5: float | None = None
    recall10: float | None = None
    precision3: float | None = None
    precision5: float | None = None
    precision10: float | None = None
    mrr: float | None = None
    ndcg3: float | None = None
    ndcg5: float | None = None
    ndcg10: float | None = None
    # Rank of first grade>=2 chunk
    first_relevant_rank: int | None = None


def evaluate_query_v2(
    query: EvalQueryV2,
    index: int,
    total: int,
    retriever: Any,
    top_k: int,
    min_grade: int = 2,
) -> QueryEvalRecordV2:
    """Run one EvalQueryV2 query and return a filled QueryEvalRecordV2."""
    from .metrics import _get_chunk_id

    error: str | None = None
    results: list[Any] = []
    try:
        results = retriever.retrieve(query.query, top_k=top_k)
    except Exception as exc:
        error = str(exc)
    diagnostics = getattr(retriever, "last_diagnostics", None)

    best_score = results[0].score if results else None

    # Unanswerable queries (OOS / invalid) skip all metric computation.
    if not query.is_answerable or not query.relevant_chunks:
        return QueryEvalRecordV2(
            index=index,
            total=total,
            query=query,
            results=results,
            error=error,
            best_score=best_score,
            diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
        )

    rc = query.relevant_chunks  # {chunk_id: grade}
    qualified_ids = {cid for cid, g in rc.items() if g >= min_grade}

    # First relevant rank (grade >= min_grade)
    first_rank: int | None = None
    for rank_i, r in enumerate(results, start=1):
        if _get_chunk_id(r) in qualified_ids:
            first_rank = rank_i
            break

    return QueryEvalRecordV2(
        index=index,
        total=total,
        query=query,
        results=results,
        error=error,
        best_score=best_score,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
        hit3=compute_hit_at_k_v2(results, rc, 3, min_grade),
        hit5=compute_hit_at_k_v2(results, rc, 5, min_grade),
        hit10=compute_hit_at_k_v2(results, rc, 10, min_grade),
        recall3=compute_recall_at_k_v2(results, rc, 3, min_grade),
        recall5=compute_recall_at_k_v2(results, rc, 5, min_grade),
        recall10=compute_recall_at_k_v2(results, rc, 10, min_grade),
        precision3=compute_precision_at_k_v2(results, rc, 3, min_grade),
        precision5=compute_precision_at_k_v2(results, rc, 5, min_grade),
        precision10=compute_precision_at_k_v2(results, rc, 10, min_grade),
        mrr=compute_mrr_v2(results, rc, min_grade),
        ndcg3=compute_ndcg_at_k(results, rc, 3),
        ndcg5=compute_ndcg_at_k(results, rc, 5),
        ndcg10=compute_ndcg_at_k(results, rc, 10),
        first_relevant_rank=first_rank,
    )


def run_evaluation_v2(
    queries: list[EvalQueryV2],
    retriever: Any,
    top_k: int,
    min_grade: int = 2,
) -> dict[str, Any]:
    """Run the V2 evaluation and return aggregate metrics.

    OOS (out_of_scope) and invalid queries are evaluated for retrieval but
    are **excluded** from all Hit/Recall/Precision/MRR/nDCG aggregates. They are
    reported separately under the ``oos_invalid`` key.

    Per-category and per-difficulty breakdowns are provided under
    ``per_category`` and ``per_difficulty``.

    Parameters
    ----------
    queries:
        List of ``EvalQueryV2`` instances (100 in the standard dataset).
    retriever:
        Any object exposing ``.retrieve(query: str, top_k: int) -> list``.
    top_k:
        Maximum results to retrieve per query.
    min_grade:
        Minimum grade to treat as "relevant" for Hit/Recall/MRR (default 2).
    """
    records: list[QueryEvalRecordV2] = []
    total = len(queries)

    # Per-category metric accumulators
    cat_hit3: dict[str, list[float]] = defaultdict(list)
    cat_hit5: dict[str, list[float]] = defaultdict(list)
    cat_hit10: dict[str, list[float]] = defaultdict(list)
    cat_recall3: dict[str, list[float]] = defaultdict(list)
    cat_recall5: dict[str, list[float]] = defaultdict(list)
    cat_recall10: dict[str, list[float]] = defaultdict(list)
    cat_precision3: dict[str, list[float]] = defaultdict(list)
    cat_precision5: dict[str, list[float]] = defaultdict(list)
    cat_precision10: dict[str, list[float]] = defaultdict(list)
    cat_mrr: dict[str, list[float]] = defaultdict(list)
    cat_ndcg3: dict[str, list[float]] = defaultdict(list)
    cat_ndcg5: dict[str, list[float]] = defaultdict(list)
    cat_ndcg10: dict[str, list[float]] = defaultdict(list)
    difficulty_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Global metric accumulators (answerable only)
    g_hit3: list[float] = []
    g_hit5: list[float] = []
    g_hit10: list[float] = []
    g_recall3: list[float] = []
    g_recall5: list[float] = []
    g_recall10: list[float] = []
    g_precision3: list[float] = []
    g_precision5: list[float] = []
    g_precision10: list[float] = []
    g_mrr: list[float] = []
    g_ndcg3: list[float] = []
    g_ndcg5: list[float] = []
    g_ndcg10: list[float] = []

    oos_invalid_records: list[QueryEvalRecordV2] = []
    unanswerable_by_category: dict[str, list[QueryEvalRecordV2]] = defaultdict(list)
    evaluated_count = 0

    for i, eq in enumerate(queries, start=1):
        rec = evaluate_query_v2(eq, i, total, retriever, top_k, min_grade)
        records.append(rec)

        # Stream progress
        print("-" * 60)
        cat_tag = f"[{eq.category.upper()}/{eq.difficulty.upper()}]"
        print(f"[{i:03d}/{total:03d}] {cat_tag} {eq.description}")
        print(f"  Q: {eq.query}")

        if not eq.is_answerable:
            oos_invalid_records.append(rec)
            unanswerable_by_category[eq.category].append(rec)

        if rec.error:
            print(f"  [ERROR] {rec.error}")
            continue

        if not eq.is_answerable:
            top1_id = None
            if rec.results:
                from .metrics import _get_chunk_id
                top1_id = _get_chunk_id(rec.results[0])
            print(f"  [UNANSWERABLE] top-1 chunk: {top1_id or 'N/A'}")
            continue

        if not eq.relevant_chunks:
            continue

        evaluated_count += 1
        cat = eq.category

        # Accumulate per-category
        cat_hit3[cat].append(rec.hit3)
        cat_hit5[cat].append(rec.hit5)
        cat_hit10[cat].append(rec.hit10)
        cat_recall3[cat].append(rec.recall3)
        cat_recall5[cat].append(rec.recall5)
        cat_recall10[cat].append(rec.recall10)
        cat_precision3[cat].append(rec.precision3)
        cat_precision5[cat].append(rec.precision5)
        cat_precision10[cat].append(rec.precision10)
        cat_mrr[cat].append(rec.mrr)
        cat_ndcg3[cat].append(rec.ndcg3)
        cat_ndcg5[cat].append(rec.ndcg5)
        cat_ndcg10[cat].append(rec.ndcg10)

        for metric_name, value in (
            ("Hit@3", rec.hit3),
            ("Hit@5", rec.hit5),
            ("Hit@10", rec.hit10),
            ("Recall@3", rec.recall3),
            ("Recall@5", rec.recall5),
            ("Recall@10", rec.recall10),
            ("Precision@3", rec.precision3),
            ("Precision@5", rec.precision5),
            ("Precision@10", rec.precision10),
            ("MRR", rec.mrr),
            ("nDCG@3", rec.ndcg3),
            ("nDCG@5", rec.ndcg5),
            ("nDCG@10", rec.ndcg10),
        ):
            if value is not None:
                difficulty_metrics[eq.difficulty][metric_name].append(value)

        # Accumulate global
        g_hit3.append(rec.hit3)
        g_hit5.append(rec.hit5)
        g_hit10.append(rec.hit10)
        g_recall3.append(rec.recall3)
        g_recall5.append(rec.recall5)
        g_recall10.append(rec.recall10)
        g_precision3.append(rec.precision3)
        g_precision5.append(rec.precision5)
        g_precision10.append(rec.precision10)
        g_mrr.append(rec.mrr)
        g_ndcg3.append(rec.ndcg3)
        g_ndcg5.append(rec.ndcg5)
        g_ndcg10.append(rec.ndcg10)

        rr_str = str(rec.first_relevant_rank) if rec.first_relevant_rank else "NOT FOUND"
        print(
            f"  nDCG@5={rec.ndcg5:.3f}  Recall@5={rec.recall5:.3f}  "
            f"MRR={rec.mrr:.3f}  First-relevant-rank={rr_str}"
        )

    # ------------------------------------------------------------------
    # Build per-category summary
    # ------------------------------------------------------------------
    def _avg(lst: list[float]) -> float | None:
        return sum(lst) / len(lst) if lst else None

    per_category: dict[str, dict[str, Any]] = {}
    all_cats = sorted(
        set(cat_hit3) | set(cat_hit5) | set(cat_hit10)
    )
    for cat in all_cats:
        per_category[cat] = {
            "count": len(cat_hit3[cat]),
            "Hit@3": _avg(cat_hit3[cat]),
            "Hit@5": _avg(cat_hit5[cat]),
            "Hit@10": _avg(cat_hit10[cat]),
            "Recall@3": _avg(cat_recall3[cat]),
            "Recall@5": _avg(cat_recall5[cat]),
            "Recall@10": _avg(cat_recall10[cat]),
            "Precision@3": _avg(cat_precision3[cat]),
            "Precision@5": _avg(cat_precision5[cat]),
            "Precision@10": _avg(cat_precision10[cat]),
            "MRR": _avg(cat_mrr[cat]),
            "nDCG@3": _avg(cat_ndcg3[cat]),
            "nDCG@5": _avg(cat_ndcg5[cat]),
            "nDCG@10": _avg(cat_ndcg10[cat]),
        }

    per_difficulty = {
        difficulty: {
            "count": len(difficulty_metrics[difficulty].get("Hit@3", [])),
            **{
                metric_name: _avg(values)
                for metric_name, values in sorted(difficulty_metrics[difficulty].items())
            },
        }
        for difficulty in ("easy", "medium", "hard")
        if difficulty in difficulty_metrics
    }

    oos_invalid = {
        category: {
            "count": len(category_records),
            "retrieval_errors": sum(bool(rec.error) for rec in category_records),
            "queries_with_results": sum(bool(rec.results) for rec in category_records),
            "records": category_records,
        }
        for category, category_records in sorted(unanswerable_by_category.items())
    }

    # ------------------------------------------------------------------
    # Build final metrics dict
    # ------------------------------------------------------------------
    valid_top1 = [rec.best_score for rec in records if rec.best_score is not None]
    avg_top1 = _avg(valid_top1)

    metrics: dict[str, Any] = {
        "evaluated_queries": evaluated_count,
        "total_queries": total,
        "unanswerable_queries": len(oos_invalid_records),
        "query_records": records,
        "oos_invalid_records": oos_invalid_records,
        "oos_invalid": oos_invalid,
        "per_category": per_category,
        "per_difficulty": per_difficulty,
        "Average_top1_similarity": avg_top1,
        "Average_top1_score": avg_top1,
        "errors": [(r.index, r.query.query, r.error) for r in records if r.error],
        "min_grade": min_grade,
    }

    if evaluated_count > 0:
        metrics["Hit@3"] = _avg(g_hit3)
        metrics["Hit@5"] = _avg(g_hit5)
        metrics["Hit@10"] = _avg(g_hit10)
        metrics["Recall@3"] = _avg(g_recall3)
        metrics["Recall@5"] = _avg(g_recall5)
        metrics["Recall@10"] = _avg(g_recall10)
        metrics["Precision@3"] = _avg(g_precision3)
        metrics["Precision@5"] = _avg(g_precision5)
        metrics["Precision@10"] = _avg(g_precision10)
        metrics["MRR"] = _avg(g_mrr)
        metrics["nDCG@3"] = _avg(g_ndcg3)
        metrics["nDCG@5"] = _avg(g_ndcg5)
        metrics["nDCG@10"] = _avg(g_ndcg10)

    return metrics
