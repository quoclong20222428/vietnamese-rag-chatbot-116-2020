"""Tests for rank fusion and the HybridRetriever composition."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval_types import RetrievalResult
from retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion
from evaluation.eval_queries_v2 import EvalQueryV2
from evaluation.reporting import render_full_report_v2
from evaluation.runner import run_evaluation_v2


def _result(chunk_id: str, rank: Any = None, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=score,
        retrieval_method="source",
        score_type="source-score",
        rank=rank,
        metadata={"article": "Điều 1"},
    )


class _StubRetriever:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = list(results or [])
        self.error = error
        self.requested_top_k: int | None = None

    def retrieve(self, query: str, top_k: int):
        self.requested_top_k = top_k
        if self.error:
            raise self.error
        return self.results[:top_k]


class TestReciprocalRankFusion(unittest.TestCase):
    def test_calculates_rrf_from_ranks(self):
        result = reciprocal_rank_fusion(
            {"hnsw": [_result("A", 1)], "bm25": [_result("A", 2)]},
            rrf_k=10,
        )[0]
        self.assertAlmostEqual(result.score, 1 / 11 + 1 / 12)
        self.assertEqual(result.score_type, "rrf")

    def test_merges_rankings_and_assigns_diagnostic_ranks(self):
        fused = reciprocal_rank_fusion(
            {
                "hnsw": [_result("A", 1), _result("B", 2), _result("C", 3)],
                "bm25": [_result("C", 1), _result("A", 2), _result("D", 3)],
            }
        )
        self.assertEqual([result.chunk_id for result in fused], ["A", "C", "B", "D"])
        self.assertAlmostEqual(fused[0].score, 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused[1].score, 1 / 61 + 1 / 63)
        self.assertEqual([result.rank for result in fused], [1, 2, 3, 4])
        self.assertEqual(
            fused[0].metadata["retrieval_diagnostics"]["source_ranks"],
            {"hnsw": 1, "bm25": 2},
        )

    def test_deduplicates_chunk_within_and_across_sources(self):
        fused = reciprocal_rank_fusion(
            {
                "hnsw": [_result("A", 1), _result("A", 2)],
                "bm25": [_result("A", 1)],
            }
        )
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].score, 2 / 61)

    def test_ignores_raw_scores_and_uses_fallback_rank_when_invalid(self):
        fused = reciprocal_rank_fusion(
            {"hnsw": [_result("A", 0, math.nan), _result("B", "bad", math.inf)]},
            rrf_k=10,
        )
        self.assertEqual([item.chunk_id for item in fused], ["A", "B"])
        self.assertAlmostEqual(fused[0].score, 1 / 11)
        self.assertAlmostEqual(fused[1].score, 1 / 12)
        self.assertTrue(all(math.isfinite(item.score) for item in fused))

    def test_top_k_limits_results(self):
        fused = reciprocal_rank_fusion(
            {"hnsw": [_result("A", 1), _result("B", 2)]}, top_k=1
        )
        self.assertEqual([item.chunk_id for item in fused], ["A"])

    def test_invalid_top_k_and_rrf_k_raise(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"hnsw": []}, top_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"hnsw": []}, rrf_k=0)


class TestHybridRetriever(unittest.TestCase):
    def test_returns_shared_contract_and_tracks_sources(self):
        retriever = HybridRetriever(
            {
                "hnsw": _StubRetriever([_result("A", 1), _result("B", 2)]),
                "bm25": _StubRetriever([_result("B", 1), _result("C", 2)]),
            },
            candidate_k=5,
        )
        results = retriever.retrieve("question", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, RetrievalResult) for result in results))
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].retrieval_method, "hybrid")
        self.assertEqual(results[0].score_type, "rrf")
        self.assertIn("retrieval_diagnostics", results[0].metadata)

    def test_handles_empty_and_single_source_failure(self):
        hnsw = _StubRetriever([_result("A", 1)])
        hybrid = HybridRetriever(
            {"hnsw": hnsw, "bm25": _StubRetriever(error=RuntimeError("offline"))}
        )
        results = hybrid.retrieve("question", top_k=5)
        self.assertEqual([item.chunk_id for item in results], ["A"])
        self.assertEqual(hybrid.last_diagnostics["errors"], {"bm25": "offline"})

        reverse = HybridRetriever(
            {
                "hnsw": _StubRetriever(error=RuntimeError("offline")),
                "bm25": _StubRetriever([_result("B", 1)]),
            }
        )
        self.assertEqual([item.chunk_id for item in reverse.retrieve("question")], ["B"])

    def test_both_empty_returns_empty(self):
        retriever = HybridRetriever(
            {"hnsw": _StubRetriever(), "bm25": _StubRetriever()}
        )
        self.assertEqual(retriever.retrieve("question"), [])

    def test_candidate_k_is_independent_of_final_top_k(self):
        hnsw = _StubRetriever([_result("A", 1), _result("B", 2)])
        bm25 = _StubRetriever([_result("C", 1), _result("D", 2)])
        retriever = HybridRetriever({"hnsw": hnsw, "bm25": bm25}, candidate_k=2)
        results = retriever.retrieve("question", top_k=4)
        self.assertEqual(hnsw.requested_top_k, 2)
        self.assertEqual(bm25.requested_top_k, 2)
        self.assertEqual(len(results), 4)

    def test_evaluation_runner_consumes_hybrid_contract_and_diagnostics(self):
        retriever = HybridRetriever(
            {
                "hnsw": _StubRetriever([_result("A", 1)]),
                "bm25": _StubRetriever([_result("A", 1)]),
            }
        )
        query = EvalQueryV2(
            query="Question",
            description="A relevant chunk",
            category="exact",
            difficulty="easy",
            is_answerable=True,
            relevant_chunks={"A": 3},
        )
        metrics = run_evaluation_v2([query], retriever, top_k=1)
        record = metrics["query_records"][0]
        self.assertEqual(metrics["Recall@3"], 1.0)
        self.assertIn("sources", record.diagnostics)
        self.assertEqual(record.results[0].score_type, "rrf")
        report = render_full_report_v2(
            query_records=metrics["query_records"],
            metrics=metrics,
            candidate_k=20,
            rrf_k=60,
        )
        self.assertIn("Hybrid Candidate Diagnostics", report)
        self.assertIn("Hybrid top-1 matched HNSW rank 1: 1/1", report)
        self.assertIn("Source ranks: hnsw=1, bm25=1", report)

        sparse_report = render_full_report_v2(
            query_records=[],
            metrics={"total_queries": 0, "evaluated_queries": 0},
            state={"total_chunks": 618},
        )
        self.assertIn("Total chunks:      618", sparse_report)
        self.assertNotIn("HNSW index:", sparse_report)

    def test_validates_configuration_and_top_k(self):
        with self.assertRaises(ValueError):
            HybridRetriever({"hnsw": _StubRetriever()}, candidate_k=0)
        retriever = HybridRetriever({"hnsw": _StubRetriever()})
        with self.assertRaises(ValueError):
            retriever.retrieve("question", top_k=0)


if __name__ == "__main__":
    unittest.main()
