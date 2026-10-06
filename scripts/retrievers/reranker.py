"""Cross-Encoder re-ranking for the RAG pipeline.

Uses a Cross-Encoder model to re-score a list of RetrievalResult objects.

Unlike a Bi-Encoder (HNSW), a Cross-Encoder processes the query and chunk together, providing more accurate relevance scores at the cost of higher computational overhead.

Usage:

```
python scripts/test_retrieval.py \
    --method hybrid \
    --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 \
    --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

"""


from __future__ import annotations

import logging
from typing import Any

if __package__ and __package__.startswith("scripts."):
    from ..retrieval_types import RetrievalResult
else:
    try:
        from retrieval_types import RetrievalResult
    except ImportError:
        from scripts.retrieval_types import RetrievalResult  # type: ignore[no-reattr]

LOGGER = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_TOP_K = 10


class CrossEncoderReranker:
    """Re-rank RetrievalResults using a Cross-Encoder.

    Parameters
    ----------
    model_name: HuggingFace model identifier. Default: ``BAAI/bge-reranker-v2-m3``.
    rerank_top_k: Number of results after re-ranking. Default: ``10``.
    max_length: Maximum token length. Default: ``512``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        max_length: int = 512,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if (
            not isinstance(rerank_top_k, int)
            or isinstance(rerank_top_k, bool)
            or rerank_top_k <= 0
        ):
            raise ValueError(
                f"rerank_top_k must be a positive integer (got {rerank_top_k!r})"
            )
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length <= 0
        ):
            raise ValueError(
                f"max_length must be a positive integer (got {max_length!r})"
            )

        self._model_name = model_name
        self._rerank_top_k = rerank_top_k
        self._max_length = max_length
        self._model: Any = None  # lazy-loaded on first call to rerank()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """HuggingFace model identifier used by this reranker."""
        return self._model_name

    @property
    def rerank_top_k(self) -> int:
        """Number of results returned after re-ranking."""
        return self._rerank_top_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Cross-Encoder model on first use (lazy initialisation)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install it with:  pip install sentence-transformers>=3.0.0"
            ) from exc

        LOGGER.info("Loading Cross-Encoder re-ranker: %s", self._model_name)
        self._model = CrossEncoder(
            self._model_name,
            max_length=self._max_length,
        )
        LOGGER.info("Cross-Encoder re-ranker loaded successfully: %s", self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Tính lại điểm và sắp xếp lại kết quả (results) theo query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty, non-whitespace string")
        if not results:
            return []

        self._load_model()

        # Build (query, passage) pairs expected by CrossEncoder.predict().
        pairs: list[tuple[str, str]] = [
            (query, result.text) for result in results
        ]

        LOGGER.debug(
            "Re-ranking %d candidates with %s ...", len(pairs), self._model_name
        )
        raw_scores: list[float] = self._model.predict(pairs).tolist()

        # Pair scores with results then sort descending.
        scored = list(zip(raw_scores, results))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep only rerank_top_k and rebuild RetrievalResult objects.
        top = scored[: self._rerank_top_k]
        reranked: list[RetrievalResult] = []
        for new_rank, (ce_score, original) in enumerate(top, start=1):
            # Copy existing metadata and inject re-ranking diagnostics.
            meta: dict[str, Any] = dict(getattr(original, "metadata", {}) or {})
            diag: dict[str, Any] = dict(meta.get("retrieval_diagnostics", {}) or {})
            diag["pre_rerank_score"] = original.score
            diag["pre_rerank_rank"] = original.rank
            diag["pre_rerank_method"] = original.retrieval_method
            diag["reranker_model"] = self._model_name
            meta["retrieval_diagnostics"] = diag

            reranked.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=float(ce_score),
                    retrieval_method="hybrid+rerank",
                    score_type="cross_encoder",
                    rank=new_rank,
                    metadata=meta,
                )
            )

        LOGGER.debug("Re-ranking complete. Returning %d results.", len(reranked))
        return reranked
