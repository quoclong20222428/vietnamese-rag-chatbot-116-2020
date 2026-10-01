"""Unit tests for scripts/retrieval.py.

Uses the same mocking strategy as tests/test_embedding.py:
- FlagEmbedding (and therefore BAAI/bge-m3) is replaced by a lightweight
  fake so the tests run without downloading the 2.3 GB model.
- psycopg is replaced by a MagicMock so no real database is needed.

Run from the repository root::

    conda activate chatbot
    python -m pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Shared fake helpers (mirrors test_embedding.py conventions)
# ---------------------------------------------------------------------------

FAKE_DIM = 1024
FAKE_VECTOR = [0.1] * FAKE_DIM


def _fake_encode(
    texts,
    batch_size=32,
    max_length=8192,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
):
    import numpy as np  # noqa: PLC0415
    return {"dense_vecs": np.ones((len(texts), FAKE_DIM), dtype="float32")}


def _make_fake_flag_embedding_module() -> types.ModuleType:
    fake_pkg = types.ModuleType("FlagEmbedding")
    fake_model_cls = MagicMock(name="BGEM3FlagModel")
    fake_model_instance = MagicMock(name="bgem3_instance")
    fake_model_instance.encode.side_effect = _fake_encode
    fake_model_cls.return_value = fake_model_instance
    fake_pkg.BGEM3FlagModel = fake_model_cls
    return fake_pkg


def _make_fake_psycopg_connect(rows=None):
    """Return a factory that yields a mock psycopg connection.

    Parameters
    ----------
    rows:
        Rows that ``cursor.fetchall()`` will return.  Defaults to a single
        row with plausible legal_chunks data.
    """
    if rows is None:
        rows = [
            (
                "chunk-001",            # chunk_id
                "Điều 1. Phạm vi điều chỉnh...",  # text
                0.92,                   # similarity
                "doc-116-2020",         # document_id
                "Nghị định 116/2020/NĐ-CP",  # document_title
                "116/2020/NĐ-CP",       # document_number
                "core",                 # source_type
                "primary",              # document_role
                "primary_legal_source", # authority_level
                100,                    # retrieval_priority
                "Chương I",             # chapter
                "Điều 1",              # article
                None,                   # clause
                None,                   # point
                "legal_text",           # content_type
            )
        ]

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_cursor.__enter__ = lambda s: mock_cursor
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)

    return mock_conn


# ---------------------------------------------------------------------------
# Fixtures — module reload helpers
# ---------------------------------------------------------------------------


def _import_retrieval():
    """Import (or re-import) the retrieval module with fakes in place."""
    for mod in ("retrieval", "embedding"):
        sys.modules.pop(mod, None)
    from retrievers import hnsw as rv  # noqa: PLC0415
    return rv


class _RetrievalTestBase(unittest.TestCase):
    """Base class that installs fakes before each test and cleans up after."""

    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------


class TestRetrieverInit(_RetrievalTestBase):
    def test_embedding_model_loaded_once(self):
        """Retriever.__init__ should load EmbeddingModel exactly once."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        # The fake BGEM3FlagModel constructor should have been called once.
        self.assertEqual(self.fake_fe.BGEM3FlagModel.call_count, 1)

    def test_model_not_reloaded_across_queries(self):
        """Calling retrieve() twice must not reload the model."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            retriever.retrieve("Điều 1 là gì?", top_k=3)
            retriever.retrieve("Điều 2 là gì?", top_k=3)

        # Constructor called once, not three times.
        self.assertEqual(self.fake_fe.BGEM3FlagModel.call_count, 1)


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation(_RetrievalTestBase):
    def _make_retriever(self, rv):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db")

    def test_empty_string_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("")

    def test_whitespace_only_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("   \t\n  ")

    def test_top_k_zero_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=0)

    def test_top_k_negative_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=-5)

    def test_top_k_exceeds_maximum_raises_value_error(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("Điều 1", top_k=rv.MAX_TOP_K + 1)

    def test_valid_query_does_not_raise(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            # Should not raise.
            results = retriever.retrieve("Điều kiện hưởng hỗ trợ là gì?", top_k=5)
            self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# 3. Query embedding
# ---------------------------------------------------------------------------


class TestQueryEmbedding(_RetrievalTestBase):
    def test_embed_texts_called_with_raw_query(self):
        """retrieve() must pass the raw query string to embed_texts."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        query = "Điều kiện để được hưởng chính sách hỗ trợ là gì?"

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            retriever.retrieve(query, top_k=3)

        model_instance = self.fake_fe.BGEM3FlagModel.return_value
        call_args = model_instance.encode.call_args
        texts_sent = call_args[0][0]  # first positional argument
        self.assertIn(query, texts_sent)

    def test_query_produces_1024d_vector(self):
        """The embedding must be 1024-dimensional before being sent to the DB."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        # Capture vectors produced by embed_query.
        original_embed = retriever._embedding_model.embed_query

        captured: list[list[float]] = []

        def _capturing_embed(text, **kwargs):
            result = original_embed(text, **kwargs)
            captured.append(result)
            return result

        retriever._embedding_model.embed_query = _capturing_embed
        fake_conn2 = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn2):
            retriever.retrieve("Điều 1 là gì?", top_k=2)

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), FAKE_DIM)


# ---------------------------------------------------------------------------
# 4. Database interaction
# ---------------------------------------------------------------------------


class TestDatabaseInteraction(_RetrievalTestBase):
    def _make_retriever(self, rv, fake_conn=None):
        if fake_conn is None:
            fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db")

    def test_returns_list_of_retrieval_results(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=5)

        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, rv.RetrievalResult) for r in results))

    def test_score_is_numeric(self):
        """Score must be a float (no type constraint on range)."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=5)

        for r in results:
            self.assertIsInstance(r.score, float)

    def test_results_ordered_by_descending_score(self):
        """Results must be ordered highest similarity first."""
        rv = _import_retrieval()
        # Provide multiple rows with decreasing similarity.
        rows = [
            ("c1", "text1", 0.95, "d1", "Title1", "N1", "core", "primary",
             "primary_legal_source", 100, "Chương I", "Điều 1", None, None, "legal_text"),
            ("c2", "text2", 0.80, "d1", "Title1", "N1", "core", "primary",
             "primary_legal_source", 100, "Chương I", "Điều 2", None, None, "legal_text"),
            ("c3", "text3", 0.65, "d1", "Title1", "N1", "core", "primary",
             "primary_legal_source", 100, "Chương I", "Điều 3", None, None, "legal_text"),
        ]
        fake_conn = _make_fake_psycopg_connect(rows=rows)
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("câu hỏi pháp lý", top_k=3)

        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_forwarded_to_db(self):
        """LIMIT parameter passed to the DB cursor must equal top_k."""
        rv = _import_retrieval()
        captured_params: list = []

        def _make_tracking_conn(rows=None):
            """Build a mock connection whose cursor.execute records every call."""
            if rows is None:
                rows = []
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = rows
            mock_cur.fetchone.return_value = None
            mock_cur.__enter__ = lambda s: mock_cur
            mock_cur.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None):
                captured_params.append((sql, params))

            mock_cur.execute.side_effect = _execute

            mock_conn = MagicMock()
            mock_conn.cursor.return_value = mock_cur
            mock_conn.__enter__ = lambda s: mock_conn
            mock_conn.__exit__ = MagicMock(return_value=False)
            return mock_conn

        # __init__ gets the first conn (for no internal DB call),
        # retrieve() gets the second.
        conns = iter([_make_tracking_conn(), _make_tracking_conn()])
        with patch.object(rv, "_connect", side_effect=lambda url: next(conns)):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
            retriever.retrieve("câu hỏi", top_k=7)

        retrieval_calls = [
            p for sql, p in captured_params
            if p is not None and "ORDER BY embedding" in sql
        ]
        self.assertTrue(len(retrieval_calls) > 0, "Retrieval SQL was not called")
        self.assertEqual(retrieval_calls[0][2], 7)

    def test_empty_db_returns_empty_list(self):
        """No results from DB should return an empty list (no exception)."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect(rows=[])
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("câu hỏi", top_k=5)

        self.assertEqual(results, [])

    def test_metadata_contains_expected_keys(self):
        """RetrievalResult.metadata must contain all schema-backed keys."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=1)

        self.assertEqual(len(results), 1)
        meta = results[0].metadata
        expected_keys = {
            "document_id",
            "document_title",
            "document_number",
            "source_type",
            "document_role",
            "authority_level",
            "retrieval_priority",
            "chapter",
            "article",
            "clause",
            "point",
            "content_type",
        }
        self.assertTrue(expected_keys.issubset(set(meta.keys())), meta.keys())

    def test_chunk_id_and_text_populated(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("Điều 1 là gì?", top_k=1)

        r = results[0]
        self.assertEqual(r.chunk_id, "chunk-001")
        self.assertIn("Điều 1", r.text)


# ---------------------------------------------------------------------------
# 5. ef_search configuration
# ---------------------------------------------------------------------------


class TestEfSearch(_RetrievalTestBase):
    def test_set_config_called_with_ef_search(self):
        """set_config for hnsw.ef_search should be called before the retrieval SQL."""
        rv = _import_retrieval()
        executed_sqls: list[str] = []

        def _make_tracking_conn():
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_cur.fetchone.return_value = None
            mock_cur.__enter__ = lambda s: mock_cur
            mock_cur.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None):
                executed_sqls.append(sql)

            mock_cur.execute.side_effect = _execute

            mock_conn = MagicMock()
            mock_conn.cursor.return_value = mock_cur
            mock_conn.__enter__ = lambda s: mock_conn
            mock_conn.__exit__ = MagicMock(return_value=False)
            return mock_conn

        conns = iter([_make_tracking_conn(), _make_tracking_conn()])
        with patch.object(rv, "_connect", side_effect=lambda url: next(conns)):
            retriever = rv.Retriever(database_url="postgresql://fake/db", ef_search=80)
            retriever.retrieve("câu hỏi", top_k=5)

        self.assertTrue(
            any("set_config" in sql for sql in executed_sqls),
            f"set_config not found in executed SQL: {executed_sqls}",
        )


# ---------------------------------------------------------------------------
# 6. verify_database_state
# ---------------------------------------------------------------------------


class TestVerifyDatabaseState(_RetrievalTestBase):
    def test_returns_expected_keys(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        # Mock the integrity check cursor responses.
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [
            (617, 617, 0),                           # _INTEGRITY_SQL
            ("legal_chunks_embedding_hnsw_idx",),    # _HNSW_INDEX_SQL
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        with patch.object(rv, "_connect", return_value=mock_conn):
            state = retriever.verify_database_state()

        self.assertEqual(state["total_chunks"], 617)
        self.assertEqual(state["embedded_chunks"], 617)
        self.assertEqual(state["missing_embeddings"], 0)
        self.assertTrue(state["hnsw_index_exists"])

    def test_detects_missing_hnsw_index(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [
            (617, 617, 0),
            None,    # HNSW index not found
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")

        with patch.object(rv, "_connect", return_value=mock_conn):
            state = retriever.verify_database_state()

        self.assertFalse(state["hnsw_index_exists"])


# ---------------------------------------------------------------------------
# 7. _vector_to_pg helper
# ---------------------------------------------------------------------------


class TestVectorToPg(unittest.TestCase):
    def setUp(self):
        self.fake_fe = _make_fake_flag_embedding_module()
        sys.modules["FlagEmbedding"] = self.fake_fe
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)

    def tearDown(self):
        sys.modules.pop("FlagEmbedding", None)
        for mod in ("retrieval", "embedding"):
            sys.modules.pop(mod, None)

    def test_pgvector_literal_format(self):
        rv = _import_retrieval()
        result = rv._vector_to_pg([0.1, 0.2, 0.3])
        self.assertTrue(result.startswith("["))
        self.assertTrue(result.endswith("]"))
        self.assertEqual(result.count(","), 2)

    def test_1024d_vector(self):
        rv = _import_retrieval()
        vec = [0.0] * 1024
        result = rv._vector_to_pg(vec)
        self.assertEqual(result.count(","), 1023)


# ---------------------------------------------------------------------------
# 8. ef_search validation
# ---------------------------------------------------------------------------


class TestEfSearchValidation(_RetrievalTestBase):
    """ef_search must be a positive integer; bool/float/string/None/<=0 are rejected."""

    def _make_retriever_with_ef(self, rv, ef_search):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db", ef_search=ef_search)

    def test_ef_search_zero_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, 0)

    def test_ef_search_negative_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, -1)

    def test_ef_search_true_raises(self):
        """True is a bool (subclass of int) and must be rejected."""
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, True)

    def test_ef_search_false_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, False)

    def test_ef_search_float_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, 1.5)

    def test_ef_search_string_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, "40")

    def test_ef_search_none_raises(self):
        rv = _import_retrieval()
        with self.assertRaises(ValueError):
            self._make_retriever_with_ef(rv, None)

    def test_ef_search_one_valid(self):
        rv = _import_retrieval()
        # Should not raise.
        retriever = self._make_retriever_with_ef(rv, 1)
        self.assertEqual(retriever._ef_search, 1)

    def test_ef_search_default_valid(self):
        rv = _import_retrieval()
        retriever = self._make_retriever_with_ef(rv, 40)
        self.assertEqual(retriever._ef_search, 40)

    def test_ef_search_large_valid(self):
        rv = _import_retrieval()
        retriever = self._make_retriever_with_ef(rv, 200)
        self.assertEqual(retriever._ef_search, 200)


# ---------------------------------------------------------------------------
# 9. Extended top_k validation (bool / float / string / None / MAX_TOP_K)
# ---------------------------------------------------------------------------


class TestTopKExtendedValidation(_RetrievalTestBase):
    """Supplement existing top_k tests with bool, float, string, None, and boundary."""

    def _make_retriever(self, rv):
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            return rv.Retriever(database_url="postgresql://fake/db")

    def test_top_k_true_raises(self):
        """True is a bool (int subclass) and must be rejected."""
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=True)

    def test_top_k_false_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=False)

    def test_top_k_float_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=1.5)

    def test_top_k_string_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k="5")

    def test_top_k_none_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=None)

    def test_top_k_one_valid(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            results = retriever.retrieve("câu hỏi", top_k=1)
        self.assertIsInstance(results, list)

    def test_top_k_max_valid(self):
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = self._make_retriever(rv)
            # Should not raise.
            retriever.retrieve("câu hỏi", top_k=rv.MAX_TOP_K)

    def test_top_k_max_plus_one_raises(self):
        rv = _import_retrieval()
        retriever = self._make_retriever(rv)
        with self.assertRaises(ValueError):
            retriever.retrieve("câu hỏi", top_k=rv.MAX_TOP_K + 1)


# ---------------------------------------------------------------------------
# 10. Device selection — mocked, no physical GPU required
# ---------------------------------------------------------------------------


class TestDeviceSelection(_RetrievalTestBase):
    """EmbeddingModel._detect_device() and EmbeddingModel.device property.

    All tests mock torch.cuda.is_available() so that no NVIDIA GPU is needed.
    """

    def _import_embedding(self):
        """Import the embedding module with the fake FlagEmbedding in place."""
        sys.modules.pop("embedding", None)
        from embeddings import embedding as em  # noqa: PLC0415
        return em

    def test_cuda_available_selects_cuda(self):
        """When torch.cuda.is_available() returns True, device must be 'cuda'."""
        em = self._import_embedding()
        with patch("torch.cuda.is_available", return_value=True):
            device = em.EmbeddingModel._detect_device()
        self.assertEqual(device, "cuda")

    def test_cuda_unavailable_selects_cpu(self):
        """When torch.cuda.is_available() returns False, device must be 'cpu'."""
        em = self._import_embedding()
        with patch("torch.cuda.is_available", return_value=False):
            device = em.EmbeddingModel._detect_device()
        self.assertEqual(device, "cpu")

    def test_torch_import_error_selects_cpu(self):
        """If torch is not importable at all, device must fall back to 'cpu'."""
        em = self._import_embedding()
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocking_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        import builtins  # noqa: PLC0415
        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            device = em._detect_device()
        self.assertEqual(device, "cpu")

    def test_device_property_reflects_detected_device(self):
        """EmbeddingModel.device must equal what _detect_device() returned."""
        em = self._import_embedding()
        # The fake BGEM3FlagModel is already in sys.modules (setUp installs it).
        # Patch _detect_device to return 'cpu' explicitly.
        with patch.object(em, "_detect_device", lambda: "cpu"):
            model = em.EmbeddingModel()
        self.assertEqual(model.device, "cpu")

    def test_retriever_device_property_delegates_to_model(self):
        """Retriever.device must return the same value as its embedding model's device."""
        rv = _import_retrieval()
        fake_conn = _make_fake_psycopg_connect()
        with patch.object(rv, "_connect", return_value=fake_conn):
            retriever = rv.Retriever(database_url="postgresql://fake/db")
        # The fake model's _detect_device returns 'cpu' (no real CUDA in tests).
        self.assertIn(retriever.device, ("cuda", "cpu"))
        # Must match the embedding model.
        self.assertEqual(retriever.device, retriever._embedding_model.device)


# ---------------------------------------------------------------------------
# 11. Evaluation metric functions
# ---------------------------------------------------------------------------


def _compute_hit_at_k(results_articles, expected, k):
    """Local copy of compute_hit_at_k for isolated metric tests."""
    if not expected:
        return 0.0
    top_k_set = {a.strip() for a in results_articles[:k] if a}
    expected_set = {e.strip() for e in expected}
    return 1.0 if top_k_set & expected_set else 0.0


def _compute_recall_at_k(results_articles, expected, k):
    """Local copy of true compute_recall_at_k for isolated metric tests."""
    if not expected:
        return 0.0
    top_k_set = {a.strip() for a in results_articles[:k] if a}
    expected_set = {e.strip() for e in expected}
    found = top_k_set & expected_set
    return len(found) / len(expected_set)


def _compute_mrr(results_articles, expected):
    """Local copy of compute_mrr for isolated metric tests."""
    if not expected:
        return 0.0
    expected_set = {e.strip() for e in expected}
    for rank, art in enumerate(results_articles, start=1):
        if art and art.strip() in expected_set:
            return 1.0 / rank
    return 0.0


class TestMetrics(unittest.TestCase):
    """Tests for Hit@K, true Recall@K, and MRR metric functions."""

    # --- Hit@K ---

    def test_hit_single_relevant_found(self):
        self.assertEqual(_compute_hit_at_k(["Điều 1", "Điều 2"], ["Điều 1"], k=5), 1.0)

    def test_hit_single_relevant_not_found(self):
        self.assertEqual(_compute_hit_at_k(["Điều 3", "Điều 4"], ["Điều 1"], k=5), 0.0)

    def test_hit_relevant_outside_k(self):
        """If the relevant article is at rank > k, Hit@K must be 0."""
        articles = ["Điều 3", "Điều 4", "Điều 5", "Điều 1"]
        self.assertEqual(_compute_hit_at_k(articles, ["Điều 1"], k=2), 0.0)
        self.assertEqual(_compute_hit_at_k(articles, ["Điều 1"], k=4), 1.0)

    def test_hit_empty_expected_returns_zero(self):
        self.assertEqual(_compute_hit_at_k(["Điều 1"], [], k=5), 0.0)

    def test_hit_multiple_expected_one_found(self):
        """Hit@K is 1.0 if ANY of the expected articles is found."""
        self.assertEqual(
            _compute_hit_at_k(["Điều 3"], ["Điều 3", "Điều 4"], k=5), 1.0
        )

    def test_hit_multiple_expected_none_found(self):
        self.assertEqual(
            _compute_hit_at_k(["Điều 5", "Điều 6"], ["Điều 3", "Điều 4"], k=5), 0.0
        )

    # --- Recall@K (true recall) ---

    def test_recall_single_relevant_found(self):
        self.assertEqual(_compute_recall_at_k(["Điều 1"], ["Điều 1"], k=5), 1.0)

    def test_recall_single_relevant_not_found(self):
        self.assertEqual(_compute_recall_at_k(["Điều 2"], ["Điều 1"], k=5), 0.0)

    def test_recall_two_expected_both_found(self):
        articles = ["Điều 1", "Điều 4", "Điều 5"]
        self.assertAlmostEqual(_compute_recall_at_k(articles, ["Điều 1", "Điều 4"], k=5), 1.0)

    def test_recall_two_expected_one_found(self):
        """Partial recall: 1 of 2 expected → 0.5."""
        articles = ["Điều 1", "Điều 5", "Điều 6"]
        self.assertAlmostEqual(_compute_recall_at_k(articles, ["Điều 1", "Điều 4"], k=5), 0.5)

    def test_recall_two_expected_none_found(self):
        self.assertAlmostEqual(
            _compute_recall_at_k(["Điều 5"], ["Điều 1", "Điều 4"], k=5), 0.0
        )

    def test_recall_empty_expected_returns_zero(self):
        self.assertEqual(_compute_recall_at_k(["Điều 1"], [], k=5), 0.0)

    def test_recall_partial_within_k(self):
        """Only articles within top-k count."""
        # relevant at ranks 1 and 5 (k=3 should find only rank-1)
        articles = ["Điều 1", "Điều 3", "Điều 6", "Điều 7", "Điều 4"]
        self.assertAlmostEqual(
            _compute_recall_at_k(articles, ["Điều 1", "Điều 4"], k=3), 0.5
        )

    # --- MRR ---

    def test_mrr_relevant_at_rank_1(self):
        self.assertAlmostEqual(_compute_mrr(["Điều 1", "Điều 2"], ["Điều 1"]), 1.0)

    def test_mrr_relevant_at_rank_2(self):
        self.assertAlmostEqual(_compute_mrr(["Điều 2", "Điều 1"], ["Điều 1"]), 0.5)

    def test_mrr_relevant_at_rank_3(self):
        self.assertAlmostEqual(
            _compute_mrr(["Điều 2", "Điều 3", "Điều 1"], ["Điều 1"]), 1 / 3
        )

    def test_mrr_no_relevant(self):
        self.assertEqual(_compute_mrr(["Điều 2", "Điều 3"], ["Điều 1"]), 0.0)

    def test_mrr_empty_expected_returns_zero(self):
        self.assertEqual(_compute_mrr(["Điều 1"], []), 0.0)

    def test_mrr_uses_first_relevant_rank(self):
        """MRR ranks the FIRST relevant hit, not the last."""
        articles = ["Điều 5", "Điều 1", "Điều 1"]
        self.assertAlmostEqual(_compute_mrr(articles, ["Điều 1"]), 0.5)

    def test_recall_equals_hit_for_single_expected(self):
        """For a single expected article, Recall@K and Hit@K must agree."""
        for k in (1, 3, 5):
            for articles in (
                ["Điều 1", "Điều 2"],
                ["Điều 3", "Điều 4"],
            ):
                h = _compute_hit_at_k(articles, ["Điều 1"], k)
                r = _compute_recall_at_k(articles, ["Điều 1"], k)
                self.assertAlmostEqual(h, r, msg=f"k={k}, articles={articles}")


# ---------------------------------------------------------------------------
# 12. Hierarchical Ground Truth Matching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
