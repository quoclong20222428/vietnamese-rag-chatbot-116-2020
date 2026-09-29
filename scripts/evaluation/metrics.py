"""Hit, recall, precision, reciprocal-rank, and nDCG metrics.

All metrics target EvalQueryV2 (chunk-ID graded relevance):

  compute_hit_at_k_v2       Binary hit: any top-k chunk has grade >= min_grade.
  compute_recall_at_k_v2    Fraction of grade >= min_grade chunks found in top-k.
  compute_precision_at_k_v2 Precision@K over unique retrieved chunks.
  compute_mrr_v2            1/rank of first chunk with grade >= min_grade.
  compute_ndcg_at_k         nDCG@K using 2^rel - 1 gain, normalised to [0, 1].

Relevance scale for V2
  3 = Highly relevant   (direct evidence)
  2 = Relevant          (supporting evidence)
  1 = Related           (context, insufficient alone)
  0 = Not relevant      (absent from relevant_chunks dict)
"""

from __future__ import annotations

import math
from typing import Any


def _get_chunk_id(result: Any) -> str | None:
    """Return the chunk_id from a retrieval result, or None if absent."""
    cid = getattr(result, "chunk_id", None)
    if cid is not None:
        return str(cid)
    meta = getattr(result, "metadata", None) or {}
    cid = meta.get("chunk_id")
    return str(cid) if cid is not None else None


def compute_hit_at_k_v2(
    results: list[Any],
    relevant_chunks: dict[str, int],
    k: int,
    min_grade: int = 2,
) -> float:
    """Binary Hit@K: 1.0 if any top-k result has grade >= *min_grade*.

    Parameters
    ----------
    results:
        List of retrieval result objects.
    relevant_chunks:
        ``{chunk_id: grade}`` mapping (grades 1-3).
    k:
        Rank cut-off.
    min_grade:
        Minimum relevance grade to count as a hit (default 2).
    """
    if k <= 0 or not relevant_chunks:
        return 0.0
    qualified = {cid for cid, g in relevant_chunks.items() if g >= min_grade}
    for r in results[:k]:
        if _get_chunk_id(r) in qualified:
            return 1.0
    return 0.0


def compute_recall_at_k_v2(
    results: list[Any],
    relevant_chunks: dict[str, int],
    k: int,
    min_grade: int = 2,
) -> float:
    """Recall@K: fraction of grade >= *min_grade* chunks found in top-k.

    Parameters
    ----------
    results:
        List of retrieval result objects.
    relevant_chunks:
        ``{chunk_id: grade}`` mapping (grades 1-3).
    k:
        Rank cut-off.
    min_grade:
        Minimum grade to include in the denominator (default 2).
    """
    if k <= 0:
        return 0.0
    qualified = {cid for cid, g in relevant_chunks.items() if g >= min_grade}
    if not qualified:
        return 0.0
    top_k_ids = {_get_chunk_id(r) for r in results[:k]}
    return len(qualified & top_k_ids) / len(qualified)


def compute_precision_at_k_v2(
    results: list[Any],
    relevant_chunks: dict[str, int],
    k: int,
    min_grade: int = 2,
) -> float:
    """Precision@K: unique grade >= *min_grade* chunks in top K, divided by K.

    The denominator stays K when fewer than K results are returned. Repeated
    chunk IDs count once so duplicates cannot inflate precision.
    """
    if k <= 0 or not relevant_chunks:
        return 0.0

    qualified = {cid for cid, grade in relevant_chunks.items() if grade >= min_grade}
    seen: set[str] = set()
    hits = 0
    for result in results[:k]:
        cid = _get_chunk_id(result)
        if cid is not None and cid not in seen:
            seen.add(cid)
            if cid in qualified:
                hits += 1
    return hits / k


def compute_mrr_v2(
    results: list[Any],
    relevant_chunks: dict[str, int],
    min_grade: int = 2,
) -> float:
    """MRR: 1/rank of the first result with grade >= *min_grade*, else 0.

    Parameters
    ----------
    results:
        List of retrieval result objects.
    relevant_chunks:
        ``{chunk_id: grade}`` mapping (grades 1-3).
    min_grade:
        Minimum grade to count as relevant (default 2).
    """
    qualified = {cid for cid, g in relevant_chunks.items() if g >= min_grade}
    if not qualified:
        return 0.0
    for rank, r in enumerate(results, start=1):
        if _get_chunk_id(r) in qualified:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    results: list[Any],
    relevant_chunks: dict[str, int],
    k: int,
) -> float:
    """nDCG@K with graded relevance.

    Gain function: ``2^rel - 1``.
    Discount:      ``1 / log2(rank + 1)``  (rank is 1-indexed).

    DCG@K  = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
    IDCG@K = DCG of the ideal ranking (top grades in descending order).
    nDCG@K = DCG@K / IDCG@K  in [0, 1].

    Out-of-scope / invalid queries (empty ``relevant_chunks``) return 0.0.
    A query with only grade-1 chunks still produces a meaningful nDCG score.

    Parameters
    ----------
    results:
        List of retrieval result objects (ranked, best first).
    relevant_chunks:
        ``{chunk_id: grade}`` mapping (grades 1-3).
    k:
        Rank cut-off.
    """
    if k <= 0 or not relevant_chunks:
        return 0.0

    # DCG@K over the retrieved ranking
    dcg = 0.0
    seen: set[str] = set()
    for i, r in enumerate(results[:k], start=1):
        cid = _get_chunk_id(r)
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        grade = relevant_chunks.get(cid, 0)
        if grade > 0:
            dcg += (2 ** grade - 1) / math.log2(i + 1)

    # IDCG@K: ideal ranking -- sort all ground-truth grades descending
    ideal_grades = sorted(relevant_chunks.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** g - 1) / math.log2(i + 1)
        for i, g in enumerate(ideal_grades, start=1)
        if g > 0
    )

    if idcg == 0.0:
        return 0.0
    # The mathematical value is in [0, 1]. Clamp tiny floating-point drift at
    # the endpoints so callers can rely on that invariant.
    return min(1.0, max(0.0, dcg / idcg))


# ---------------------------------------------------------------------------
