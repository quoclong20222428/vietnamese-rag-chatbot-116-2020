"""Unit tests for scripts/generator/gemini.py and scripts/rag_pipeline.py.

All Gemini API calls are mocked.  Tests require no real API key, incur no
API cost, and make no network requests.

Run from the repository root::

    conda activate chatbot
    python -m pytest tests/test_generation.py -v
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make scripts/ importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval_types import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers: minimal fakes shared across test classes
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str = "chunk-1",
    text: str = "Giáo viên phải có đủ năm học kinh nghiệm.",
    score: float = 0.9,
    retrieval_method: str = "hybrid",
    rank: int = 1,
    metadata: dict | None = None,
) -> RetrievalResult:
    """Return a minimal RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        retrieval_method=retrieval_method,
        score_type="rrf",
        rank=rank,
        metadata=metadata or {
            "document_title": "Nghị định 116/2020/NĐ-CP",
            "document_number": "116/2020/NĐ-CP",
            "document_role": "primary",
            "chapter": "Chương II",
            "article": "Điều 5",
            "clause": "Khoản 2",
            "point": "Điểm a",
        },
    )


def _make_fake_genai_module(response_text: str = "Câu trả lời giả.") -> types.ModuleType:
    """Build a minimal fake google.genai module."""
    fake_genai = types.ModuleType("google.genai")
    fake_types = types.ModuleType("google.genai.types")

    mock_config_cls = MagicMock(name="GenerateContentConfig")
    fake_types.GenerateContentConfig = mock_config_cls

    mock_response = MagicMock()
    mock_response.text = response_text

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    fake_genai.Client = MagicMock(return_value=mock_client)

    # Always replace the full google namespace so each call installs fresh mocks.
    # Using setdefault would keep the first stub and leak side_effects across tests.
    google_pkg = types.ModuleType("google")
    google_pkg.genai = fake_genai
    sys.modules["google"] = google_pkg
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_types

    return fake_genai


class _StubRetriever:
    """Stub retriever for pipeline tests."""

    def __init__(self, results=None, error: Exception | None = None):
        self.results = list(results or [])
        self.error = error
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def retrieve(self, query: str, top_k: int) -> list:
        self.last_query = query
        self.last_top_k = top_k
        if self.error:
            raise self.error
        return self.results[:top_k]


class _StubGenerator:
    """Stub generator for pipeline tests."""

    def __init__(self, answer: str = "Đây là câu trả lời.", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.last_system_instruction: str | None = None
        self.last_user_prompt: str | None = None

    def generate(self, *, system_instruction: str, user_prompt: str, **kwargs) -> str:
        self.last_system_instruction = system_instruction
        self.last_user_prompt = user_prompt
        if self.error:
            raise self.error
        return self.answer


# ===========================================================================
# GeminiClient tests
# ===========================================================================


class TestGeminiClientConfiguration(unittest.TestCase):
    """Verify environment loading and missing-config error handling."""

    def setUp(self):
        # Ensure fake genai module is in sys.modules before importing GeminiClient.
        _make_fake_genai_module()
        # Clean up any cached module import from previous tests.
        for key in list(sys.modules):
            if "generator.gemini" in key or "scripts.generator.gemini" in key:
                del sys.modules[key]

    def test_raises_when_api_key_missing(self):
        """Missing GEMINI_API_KEY produces a clear RuntimeError."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "GEMINI_MODEL")}
        env["GEMINI_MODEL"] = "gemini-2.0-flash"
        with patch.dict(os.environ, env, clear=True):
            # Patch load_dotenv so it does not load the real .env.
            with patch("environment.load_dotenv"):
                from generator.gemini import GeminiClient
                with self.assertRaises(RuntimeError) as ctx:
                    GeminiClient()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_raises_when_model_missing(self):
        """Missing GEMINI_MODEL produces a clear RuntimeError."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "GEMINI_MODEL")}
        env["GEMINI_API_KEY"] = "fake-key"
        with patch.dict(os.environ, env, clear=True):
            with patch("environment.load_dotenv"):
                from generator.gemini import GeminiClient
                with self.assertRaises(RuntimeError) as ctx:
                    GeminiClient()
        self.assertIn("GEMINI_MODEL", str(ctx.exception))

    def test_accepts_explicit_api_key_and_model(self):
        """Constructor keyword arguments bypass environment lookup."""
        with patch("environment.load_dotenv"):
            from generator.gemini import GeminiClient
            client = GeminiClient(api_key="explicit-key", model="gemini-2.0-flash")
        self.assertEqual(client.model, "gemini-2.0-flash")

    def test_raises_import_error_when_sdk_missing(self):
        """Missing google-genai raises ImportError with install hint."""
        # Removing from sys.modules is insufficient once the package is installed
        # because Python would re-import it from the filesystem.  Instead, patch
        # the built-in __import__ to raise ImportError for 'google.genai' only.
        import builtins  # noqa: PLC0415
        original_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name in ("google.genai", "google"):
                raise ImportError("No module named 'google.genai'")
            return original_import(name, *args, **kwargs)

        for key in list(sys.modules):
            if "generator.gemini" in key:
                del sys.modules[key]

        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GEMINI_MODEL": "m"}):
                with patch("environment.load_dotenv"):
                    from generator.gemini import GeminiClient
                    with self.assertRaises((ImportError, RuntimeError)) as ctx:
                        GeminiClient()
        self.assertIn("google-genai", str(ctx.exception))




class TestGeminiClientGenerate(unittest.TestCase):
    """Verify generate() behaviour: success, empty response, and API failure."""

    def _make_client(self, response_text="Câu trả lời."):
        """Return a GeminiClient with a fake SDK."""
        _make_fake_genai_module(response_text)
        for key in list(sys.modules):
            if "generator.gemini" in key:
                del sys.modules[key]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GEMINI_MODEL": "gemini-2.0-flash"}):
            with patch("environment.load_dotenv"):
                from generator.gemini import GeminiClient
                return GeminiClient()

    def test_returns_text_on_success(self):
        client = self._make_client("Câu trả lời hợp lệ.")
        result = client.generate(
            system_instruction="system",
            user_prompt="question",
        )
        self.assertEqual(result, "Câu trả lời hợp lệ.")

    def test_raises_on_empty_response(self):
        """An empty model response raises RuntimeError."""
        _make_fake_genai_module("")
        for key in list(sys.modules):
            if "generator.gemini" in key:
                del sys.modules[key]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GEMINI_MODEL": "m"}):
            with patch("environment.load_dotenv"):
                from generator.gemini import GeminiClient
                # Patch _extract_text to return empty string to simulate empty response.
                with patch.object(GeminiClient, "_extract_text", return_value=""):
                    client = GeminiClient()
                    with self.assertRaises(RuntimeError) as ctx:
                        client.generate(system_instruction="s", user_prompt="q")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_raises_on_api_exception(self):
        """API exceptions are wrapped in RuntimeError."""
        _make_fake_genai_module()
        for key in list(sys.modules):
            if "generator.gemini" in key:
                del sys.modules[key]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GEMINI_MODEL": "m"}):
            with patch("environment.load_dotenv"):
                from generator.gemini import GeminiClient
                client = GeminiClient()
                # Use patch.object so the side_effect is scoped to this test only
                # and does not leak into subsequent tests via the shared mock.
                with patch.object(
                    client._client.models,
                    "generate_content",
                    side_effect=Exception("quota exceeded"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        client.generate(system_instruction="s", user_prompt="q")
        self.assertIn("quota exceeded", str(ctx.exception))

    def test_raises_on_empty_system_instruction(self):
        client = self._make_client()
        with self.assertRaises(ValueError):
            client.generate(system_instruction="", user_prompt="q")

    def test_raises_on_empty_user_prompt(self):
        client = self._make_client()
        with self.assertRaises(ValueError):
            client.generate(system_instruction="s", user_prompt="")


# ===========================================================================
# Evidence formatting tests
# ===========================================================================


class TestFormatEvidenceBlocks(unittest.TestCase):
    """Verify context construction and source metadata extraction."""

    def setUp(self):
        from rag_pipeline import format_evidence_blocks
        self.format = format_evidence_blocks

    def test_single_result_produces_one_block_with_label(self):
        results = [_make_result()]
        context, sources = self.format(results)
        self.assertIn("[S1]", context)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["label"], "[S1]")

    def test_multiple_results_have_distinct_labels(self):
        results = [
            _make_result("c1", "Văn bản A"),
            _make_result("c2", "Văn bản B", rank=2),
        ]
        context, sources = self.format(results)
        self.assertIn("[S1]", context)
        self.assertIn("[S2]", context)
        self.assertEqual(sources[0]["label"], "[S1]")
        self.assertEqual(sources[1]["label"], "[S2]")

    def test_source_label_to_content_mapping_is_stable(self):
        """Each source label maps to the correct chunk."""
        r1 = _make_result("chunk-A", "Nội dung A")
        r2 = _make_result("chunk-B", "Nội dung B", rank=2)
        context, sources = self.format([r1, r2])
        self.assertEqual(sources[0]["chunk_id"], "chunk-A")
        self.assertEqual(sources[1]["chunk_id"], "chunk-B")
        # Verify the text appears near its label in the context block.
        idx_s1 = context.index("[S1]")
        idx_s2 = context.index("[S2]")
        self.assertIn("Nội dung A", context[idx_s1:idx_s2])
        self.assertIn("Nội dung B", context[idx_s2:])

    def test_citation_hint_uses_available_legal_structure(self):
        results = [_make_result()]
        _, sources = self.format(results)
        hint = sources[0]["citation_hint"]
        # Should contain the article and document title from the metadata.
        self.assertIn("Điều 5", hint)
        self.assertIn("Nghị định 116/2020/NĐ-CP", hint)

    def test_citation_hint_omits_missing_fields(self):
        """Missing article/clause/point are NOT fabricated."""
        result = _make_result(metadata={
            "document_title": "Nghị định 116/2020/NĐ-CP",
            "document_number": "116/2020/NĐ-CP",
            # article, clause, point deliberately absent.
        })
        _, sources = self.format([result])
        hint = sources[0]["citation_hint"]
        # Citation hint should not contain fabricated article numbers.
        self.assertNotIn("Điều", hint)
        self.assertNotIn("Khoản", hint)
        # But document title is present.
        self.assertIn("Nghị định 116/2020/NĐ-CP", hint)

    def test_empty_results_produce_empty_output(self):
        context, sources = self.format([])
        self.assertEqual(context, "")
        self.assertEqual(sources, [])


# ===========================================================================
# RAGPipeline tests
# ===========================================================================


class TestRAGPipelineQuery(unittest.TestCase):
    """Verify pipeline orchestration, error handling, and response structure."""

    def _make_pipeline(self, results=None, gen_answer="Đây là câu trả lời.", top_k=5):
        from rag_pipeline import RAGPipeline
        retriever = _StubRetriever(results=results or [_make_result()])
        generator = _StubGenerator(answer=gen_answer)
        return RAGPipeline(retriever=retriever, generator=generator, top_k=top_k)

    def test_raises_on_empty_question(self):
        pipeline = self._make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.query("")

    def test_raises_on_whitespace_question(self):
        pipeline = self._make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.query("   ")

    def test_returns_rag_response_with_answer_and_sources(self):
        from rag_pipeline import RAGResponse
        pipeline = self._make_pipeline()
        response = pipeline.query("Giáo viên cần điều kiện gì?")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsInstance(response.answer, str)
        self.assertTrue(response.answer)
        self.assertIsInstance(response.sources, list)
        self.assertGreater(len(response.sources), 0)

    def test_retriever_receives_correct_top_k(self):
        from rag_pipeline import RAGPipeline
        stub = _StubRetriever(results=[_make_result()])
        pipeline = RAGPipeline(retriever=stub, generator=_StubGenerator(), top_k=3)
        pipeline.query("câu hỏi")
        self.assertEqual(stub.last_top_k, 3)

    def test_retriever_receives_original_question(self):
        from rag_pipeline import RAGPipeline
        stub = _StubRetriever(results=[_make_result()])
        pipeline = RAGPipeline(retriever=stub, generator=_StubGenerator())
        pipeline.query("Điều kiện học bổng sư phạm là gì?")
        self.assertEqual(stub.last_query, "Điều kiện học bổng sư phạm là gì?")

    def test_system_prompt_and_evidence_passed_to_generator(self):
        from rag_pipeline import RAGPipeline, SYSTEM_PROMPT
        gen = _StubGenerator()
        pipeline = RAGPipeline(
            retriever=_StubRetriever(results=[_make_result()]),
            generator=gen,
        )
        pipeline.query("câu hỏi pháp lý")
        # System prompt must be the Vietnamese legal prompt.
        self.assertEqual(gen.last_system_instruction, SYSTEM_PROMPT)
        # User prompt must contain evidence blocks and the question.
        self.assertIn("[S1]", gen.last_user_prompt)
        self.assertIn("câu hỏi pháp lý", gen.last_user_prompt)

    def test_empty_retrieval_skips_generation(self):
        from rag_pipeline import RAGPipeline
        gen = _StubGenerator()
        pipeline = RAGPipeline(
            retriever=_StubRetriever(results=[]),
            generator=gen,
        )
        response = pipeline.query("câu hỏi")
        # Generator must not have been called.
        self.assertIsNone(gen.last_user_prompt)
        self.assertEqual(response.retrieved_count, 0)
        # Answer should communicate insufficient information.
        self.assertTrue(response.answer)

    def test_retrieval_failure_raises_runtime_error(self):
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(
            retriever=_StubRetriever(error=ConnectionError("db offline")),
            generator=_StubGenerator(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.query("câu hỏi")
        self.assertIn("Retrieval failed", str(ctx.exception))

    def test_generation_failure_raises_runtime_error(self):
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(
            retriever=_StubRetriever(results=[_make_result()]),
            generator=_StubGenerator(error=RuntimeError("quota")),
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.query("câu hỏi")
        self.assertIn("Generation failed", str(ctx.exception))

    def test_retrieval_and_generation_failures_are_distinguishable(self):
        """The error message indicates which stage failed."""
        from rag_pipeline import RAGPipeline
        retrieval_err = None
        try:
            RAGPipeline(
                retriever=_StubRetriever(error=ValueError("bad query")),
                generator=_StubGenerator(),
            ).query("câu hỏi")
        except RuntimeError as exc:
            retrieval_err = str(exc)

        gen_err = None
        try:
            RAGPipeline(
                retriever=_StubRetriever(results=[_make_result()]),
                generator=_StubGenerator(error=RuntimeError("model error")),
            ).query("câu hỏi")
        except RuntimeError as exc:
            gen_err = str(exc)

        self.assertIsNotNone(retrieval_err)
        self.assertIsNotNone(gen_err)
        self.assertIn("Retrieval", retrieval_err)
        self.assertIn("Generation", gen_err)

    def test_response_has_consistent_structure(self):
        """RAGResponse always has answer, sources, retrieval_method, retrieved_count."""
        from rag_pipeline import RAGPipeline, RAGResponse
        pipeline = RAGPipeline(
            retriever=_StubRetriever(results=[_make_result()]),
            generator=_StubGenerator(),
        )
        response = pipeline.query("câu hỏi")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsNotNone(response.answer)
        self.assertIsNotNone(response.sources)
        self.assertIsNotNone(response.retrieval_method)
        self.assertIsNotNone(response.retrieved_count)

    def test_multiple_results_retain_correct_source_labels(self):
        """Source labels [S1], [S2], … map to their respective chunks."""
        from rag_pipeline import RAGPipeline
        results = [
            _make_result("chunk-A", "Nội dung A", rank=1),
            _make_result("chunk-B", "Nội dung B", rank=2),
            _make_result("chunk-C", "Nội dung C", rank=3),
        ]
        pipeline = RAGPipeline(
            retriever=_StubRetriever(results=results),
            generator=_StubGenerator(),
            top_k=3,
        )
        response = pipeline.query("câu hỏi")
        self.assertEqual(len(response.sources), 3)
        self.assertEqual(response.sources[0]["label"], "[S1]")
        self.assertEqual(response.sources[0]["chunk_id"], "chunk-A")
        self.assertEqual(response.sources[1]["label"], "[S2]")
        self.assertEqual(response.sources[1]["chunk_id"], "chunk-B")
        self.assertEqual(response.sources[2]["label"], "[S3]")
        self.assertEqual(response.sources[2]["chunk_id"], "chunk-C")

    def test_constructor_validates_retriever_interface(self):
        from rag_pipeline import RAGPipeline
        with self.assertRaises(TypeError):
            RAGPipeline(retriever=object(), generator=_StubGenerator())

    def test_constructor_validates_generator_interface(self):
        from rag_pipeline import RAGPipeline
        with self.assertRaises(TypeError):
            RAGPipeline(retriever=_StubRetriever(), generator=object())

    def test_constructor_validates_top_k(self):
        from rag_pipeline import RAGPipeline
        with self.assertRaises(ValueError):
            RAGPipeline(retriever=_StubRetriever(), generator=_StubGenerator(), top_k=0)


if __name__ == "__main__":
    unittest.main()
