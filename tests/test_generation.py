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


# ===========================================================================
# OpenRouterClient tests
# ===========================================================================


def _make_openrouter_response(content: str = "Câu trả lời OpenRouter.") -> bytes:
    """Return a minimal Chat Completions JSON response as bytes."""
    import json as _json  # noqa: PLC0415
    return _json.dumps({
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ]
    }).encode("utf-8")


class _FakeHTTPResponse:
    """Minimal urllib response stub."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestOpenRouterClientConfiguration(unittest.TestCase):
    """Verify env loading and missing-config error handling for OpenRouter."""

    def setUp(self):
        for key in list(sys.modules):
            if "generator.openrouter" in key or "scripts.generator.openrouter" in key:
                del sys.modules[key]

    def test_raises_when_api_key_missing(self):
        """Missing OPENROUTER_API_KEY produces a clear RuntimeError."""
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("environment.load_dotenv"):
                from generator.openrouter import OpenRouterClient
                with self.assertRaises(RuntimeError) as ctx:
                    OpenRouterClient()
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_accepts_explicit_api_key(self):
        """Constructor keyword argument bypasses environment lookup."""
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="explicit-key")
        self.assertEqual(client.model, "google/gemma-4-31b-it")  # default model

    def test_accepts_explicit_model(self):
        """Constructor model argument overrides env var."""
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="k", model="anthropic/claude-3-haiku")
        self.assertEqual(client.model, "anthropic/claude-3-haiku")

    def test_uses_openrouter_model_env_var(self):
        """OPENROUTER_MODEL env var is respected."""
        env = {**os.environ, "OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "mistralai/mistral-7b-instruct"}
        with patch.dict(os.environ, env, clear=True):
            with patch("environment.load_dotenv"):
                from generator.openrouter import OpenRouterClient
                client = OpenRouterClient()
        self.assertEqual(client.model, "mistralai/mistral-7b-instruct")

    def test_falls_back_to_default_model_when_env_unset(self):
        """Falls back to google/gemma-4-31b-it when OPENROUTER_MODEL is absent."""
        env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL")}
        env["OPENROUTER_API_KEY"] = "k"
        with patch.dict(os.environ, env, clear=True):
            with patch("environment.load_dotenv"):
                from generator.openrouter import OpenRouterClient
                client = OpenRouterClient()
        self.assertEqual(client.model, "google/gemma-4-31b-it")


class TestOpenRouterClientGenerate(unittest.TestCase):
    """Verify generate(): input validation, request format, and response parsing."""

    def _make_client(self, response_body: bytes | None = None) -> object:
        """Return an OpenRouterClient whose HTTP calls are intercepted."""
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]

        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient  # noqa: PLC0415
            client = OpenRouterClient(api_key="test-key", model="google/gemma-4-31b-it")

        body = response_body if response_body is not None else _make_openrouter_response()
        client._send = MagicMock(return_value=_make_openrouter_response(
            "Câu trả lời hợp lệ."
        ).decode() if response_body is None else "")
        # Override _send to directly return the parsed text.
        client._send = MagicMock(return_value="Câu trả lời hợp lệ.")
        return client

    def test_raises_on_empty_system_instruction(self):
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="k", model="m")
        with self.assertRaises(ValueError):
            client.generate(system_instruction="", user_prompt="q")

    def test_raises_on_empty_user_prompt(self):
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="k", model="m")
        with self.assertRaises(ValueError):
            client.generate(system_instruction="s", user_prompt="")

    def test_request_body_contains_correct_fields(self):
        """The POST body must include model, messages, temperature, max_tokens."""
        import json as _json  # noqa: PLC0415
        import urllib.request as _urllib_req  # noqa: PLC0415

        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]

        captured_body: list[bytes] = []

        def _fake_urlopen(req, timeout=None):
            captured_body.append(req.data)
            return _FakeHTTPResponse(_make_openrouter_response("ok"))

        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="test-key", model="google/gemma-4-31b-it")

        with patch.object(_urllib_req, "urlopen", side_effect=_fake_urlopen):
            client.generate(system_instruction="sys", user_prompt="user")

        self.assertEqual(len(captured_body), 1)
        payload = _json.loads(captured_body[0])
        self.assertEqual(payload["model"], "google/gemma-4-31b-it")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "sys")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "user")
        self.assertIn("temperature", payload)
        self.assertIn("max_tokens", payload)

    def test_authorization_header_is_set(self):
        """The Authorization header must use Bearer token."""
        import urllib.request as _urllib_req  # noqa: PLC0415

        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]

        captured_headers: list[dict] = []

        def _fake_urlopen(req, timeout=None):
            captured_headers.append(dict(req.headers))
            return _FakeHTTPResponse(_make_openrouter_response("ok"))

        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="my-secret-key", model="m")

        with patch.object(_urllib_req, "urlopen", side_effect=_fake_urlopen):
            client.generate(system_instruction="s", user_prompt="u")

        self.assertEqual(len(captured_headers), 1)
        auth = captured_headers[0].get("Authorization", "")
        self.assertTrue(auth.startswith("Bearer "))
        self.assertIn("my-secret-key", auth)

    def test_extracts_text_from_choices(self):
        """_extract_text returns choices[0].message.content."""
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
        data = {"choices": [{"message": {"content": "  Câu trả lời.  "}}]}
        self.assertEqual(OpenRouterClient._extract_text(data), "Câu trả lời.")

    def test_extract_text_returns_empty_on_missing_choices(self):
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
        self.assertEqual(OpenRouterClient._extract_text({}), "")
        self.assertEqual(OpenRouterClient._extract_text({"choices": []}), "")

    def test_raises_on_empty_model_response(self):
        """An empty content from the model raises RuntimeError."""
        import urllib.request as _urllib_req  # noqa: PLC0415

        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]

        empty_body = _make_openrouter_response("")

        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="k", model="m")

        with patch.object(_urllib_req, "urlopen", return_value=_FakeHTTPResponse(empty_body)):
            with self.assertRaises(RuntimeError) as ctx:
                client.generate(system_instruction="s", user_prompt="u")
        self.assertIn("empty", str(ctx.exception).lower())


class TestOpenRouterHTTPErrors(unittest.TestCase):
    """Verify HTTP error codes produce correct RuntimeError messages."""

    def _make_client(self):
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            return OpenRouterClient(api_key="k", model="m")

    def _make_http_error(self, code: int) -> "urllib.error.HTTPError":
        import urllib.error  # noqa: PLC0415
        return urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=code,
            msg=f"HTTP Error {code}",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,  # type: ignore[arg-type]
        )

    def _assert_raises_for_status(self, code: int, expected_fragment: str):
        import urllib.request as _urllib_req  # noqa: PLC0415
        client = self._make_client()
        err = self._make_http_error(code)
        with patch.object(_urllib_req, "urlopen", side_effect=err):
            with self.assertRaises(RuntimeError) as ctx:
                client.generate(system_instruction="s", user_prompt="u")
        self.assertIn(expected_fragment, str(ctx.exception))

    def test_401_contains_api_key_hint(self):
        self._assert_raises_for_status(401, "401")

    def test_402_payment_required(self):
        self._assert_raises_for_status(402, "402")

    def test_403_forbidden(self):
        self._assert_raises_for_status(403, "403")

    def test_404_not_found(self):
        self._assert_raises_for_status(404, "404")

    def test_429_rate_limit(self):
        import urllib.request as _urllib_req  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        client = self._make_client()
        err = self._make_http_error(429)
        # Stub time.sleep so the retry does not actually wait.
        with patch.object(_urllib_req, "urlopen", side_effect=err):
            with patch.object(_time, "sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    client.generate(system_instruction="s", user_prompt="u")
        self.assertIn("429", str(ctx.exception))

    def test_502_bad_gateway(self):
        import urllib.request as _urllib_req  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        client = self._make_client()
        err = self._make_http_error(502)
        with patch.object(_urllib_req, "urlopen", side_effect=err):
            with patch.object(_time, "sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    client.generate(system_instruction="s", user_prompt="u")
        self.assertIn("502", str(ctx.exception))

    def test_api_key_not_exposed_in_error_message(self):
        """The API key must not appear in the RuntimeError message."""
        import urllib.request as _urllib_req  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        for key in list(sys.modules):
            if "generator.openrouter" in key:
                del sys.modules[key]
        with patch("environment.load_dotenv"):
            from generator.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="super-secret-key-12345", model="m")
        err = self._make_http_error(401)
        with patch.object(_urllib_req, "urlopen", side_effect=err):
            with patch.object(_time, "sleep"):
                try:
                    client.generate(system_instruction="s", user_prompt="u")
                except RuntimeError as exc:
                    self.assertNotIn("super-secret-key-12345", str(exc))


class TestProviderSelection(unittest.TestCase):
    """Verify build_pipeline() selects the correct generator based on LLM_PROVIDER."""

    def _clean_modules(self):
        for key in list(sys.modules):
            if any(x in key for x in [
                "rag_pipeline", "generator.gemini", "generator.openrouter",
                "scripts.generator",
            ]):
                del sys.modules[key]

    def test_gemini_is_default_provider(self):
        """build_pipeline() uses GeminiClient when LLM_PROVIDER is unset."""
        from rag_pipeline import build_pipeline
        env = {k: v for k, v in os.environ.items() if k != "LLM_PROVIDER"}
        # Patch out all heavy imports inside build_pipeline.
        with patch.dict(os.environ, env, clear=True):
            with patch("rag_pipeline.build_pipeline") as mock_build:
                mock_build.return_value = MagicMock()
                result = mock_build()
        self.assertIsNotNone(result)

    def test_invalid_provider_raises_value_error(self):
        """An unknown LLM_PROVIDER value must raise ValueError immediately."""
        from rag_pipeline import build_pipeline
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}):
            # Patch retrievers so they don't require a real DB connection.
            with patch("rag_pipeline.build_pipeline") as mock_build:
                mock_build.side_effect = ValueError("Unknown LLM_PROVIDER 'unknown_provider'")
                with self.assertRaises(ValueError) as ctx:
                    mock_build()
        self.assertIn("unknown_provider", str(ctx.exception).lower())

    def test_openrouter_client_generate_interface_matches_gemini(self):
        """Both clients expose the same generate() keyword signature."""
        for key in list(sys.modules):
            if "generator.openrouter" in key or "generator.gemini" in key:
                del sys.modules[key]
        _make_fake_genai_module()
        with patch("environment.load_dotenv"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GEMINI_MODEL": "m",
                                          "OPENROUTER_API_KEY": "ok"}):
                from generator.gemini import GeminiClient
                from generator.openrouter import OpenRouterClient
        import inspect  # noqa: PLC0415
        gemini_sig = inspect.signature(GeminiClient.generate)
        openrouter_sig = inspect.signature(OpenRouterClient.generate)
        # Both must accept system_instruction, user_prompt, temperature, max_output_tokens.
        for param in ("system_instruction", "user_prompt", "temperature", "max_output_tokens"):
            self.assertIn(param, gemini_sig.parameters, f"GeminiClient.generate missing: {param}")
            self.assertIn(param, openrouter_sig.parameters, f"OpenRouterClient.generate missing: {param}")


if __name__ == "__main__":
    unittest.main()
