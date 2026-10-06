"""Rank-fusion retrieval over independent retriever implementations."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

if __package__.startswith("scripts."):
    from ..retrieval_types import RetrievalResult
else:
    from retrieval_types import RetrievalResult

LOGGER = logging.getLogger(__name__)

DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_TOP_K = 10
DEFAULT_RRF_K = 60
MAX_TOP_K = 1000


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer (got {value!r})")
    return value


def _rank(result: Any, fallback: int) -> int:
    rank = getattr(result, "rank", None)
    if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
        return rank
    return fallback


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Any]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse ranked result lists without comparing their raw scores."""
    rrf_k = _positive_int(rrf_k, "rrf_k")
    if top_k is not None:
        _positive_int(top_k, "top_k")
        if top_k > MAX_TOP_K:
            raise ValueError(f"top_k={top_k} exceeds the maximum allowed value ({MAX_TOP_K})")

    fused: dict[str, dict[str, Any]] = {}
    first_seen = 0

    for retriever_name, results in rankings.items():
        seen: set[str] = set()
        for position, result in enumerate(results, start=1):
            chunk_id = getattr(result, "chunk_id", None)
            if chunk_id is None or not str(chunk_id).strip():
                continue
            chunk_id = str(chunk_id)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            result_rank = _rank(result, position)
            entry = fused.get(chunk_id)
            if entry is None:
                entry = {
                    "result": result,
                    "score": 0.0,
                    "source_ranks": {},
                    "first_seen": first_seen,
                }
                fused[chunk_id] = entry
                first_seen += 1
            entry["score"] += 1.0 / (rrf_k + result_rank)
            entry["source_ranks"][str(retriever_name)] = result_rank

    ordered = sorted(
        fused.items(),
        key=lambda item: (-item[1]["score"], item[1]["first_seen"]),
    )
    if top_k is not None:
        ordered = ordered[:top_k]

    source_names = [str(name) for name in rankings]
    output: list[RetrievalResult] = []
    for final_rank, (chunk_id, entry) in enumerate(ordered, start=1):
        original = entry["result"]
        metadata = getattr(original, "metadata", None)
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        source_ranks = {
            name: entry["source_ranks"].get(name) for name in source_names
        }
        metadata["retrieval_diagnostics"] = {
            "rrf_score": entry["score"],
            "source_ranks": source_ranks,
        }
        score = float(entry["score"])
        if not math.isfinite(score):
            continue
        output.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=str(getattr(original, "text", "") or ""),
                score=score,
                retrieval_method="hybrid",
                score_type="rrf",
                rank=final_rank,
                metadata=metadata,
            )
        )
    return output


class HybridRetriever:
    """Retrieve candidates from multiple sources and fuse their rankings."""

    def __init__(
        self,
        retrievers: Mapping[str, Any],
        *,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not retrievers:
            raise ValueError("retrievers must contain at least one retriever")
        if any(not isinstance(name, str) or not name.strip() for name in retrievers):
            raise ValueError("retriever names must be non-empty strings")
        self._retrievers = dict(retrievers)
        self._candidate_k = _positive_int(candidate_k, "candidate_k")
        if self._candidate_k > MAX_TOP_K:
            raise ValueError(
                f"candidate_k={self._candidate_k} exceeds the maximum allowed value ({MAX_TOP_K})"
            )
        self._rrf_k = _positive_int(rrf_k, "rrf_k")
        self._last_diagnostics: dict[str, Any] = {}

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    @property
    def last_diagnostics(self) -> dict[str, Any]:
        return self._last_diagnostics

    def retrieve(
        self, query: str, top_k: int = DEFAULT_FINAL_TOP_K
    ) -> list[RetrievalResult]:
        """Return at most *top_k* fused results for a query."""
        self._last_diagnostics = {}
        top_k = _positive_int(top_k, "top_k")
        if top_k > MAX_TOP_K:
            raise ValueError(f"top_k={top_k} exceeds the maximum allowed value ({MAX_TOP_K})")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty, non-whitespace string")

        rankings: dict[str, Sequence[Any]] = {}
        sources: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for name, retriever in self._retrievers.items():
            try:
                results = retriever.retrieve(query, top_k=self._candidate_k)
                results = results if isinstance(results, Sequence) else []
            except Exception as exc:
                LOGGER.warning("%s retrieval failed: %s", name, exc)
                results = []
                errors[name] = str(exc)

            rankings[name] = results
            source_candidates: list[dict[str, Any]] = []
            seen: set[str] = set()
            for position, result in enumerate(results, start=1):
                chunk_id = getattr(result, "chunk_id", None)
                if chunk_id is None or not str(chunk_id).strip():
                    continue
                chunk_id = str(chunk_id)
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                source_candidates.append(
                    {"chunk_id": chunk_id, "rank": _rank(result, position)}
                )
            sources[name] = {
                "count": len(source_candidates),
                "candidates": source_candidates,
            }

        fused = reciprocal_rank_fusion(rankings, rrf_k=self._rrf_k, top_k=top_k)
        self._last_diagnostics = {
            "candidate_k": self._candidate_k,
            "final_top_k": top_k,
            "rrf_k": self._rrf_k,
            "sources": sources,
            "errors": errors,
        }
        return fused
