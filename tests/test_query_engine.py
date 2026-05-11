"""Unit tests for src/query_engine.py — DB and preprocessor calls are mocked."""

import pytest
from unittest.mock import MagicMock, patch

from src import query_engine
from src.similarity import cosine_similarity as _real_cosine


# ── Helpers ──────────────────────────────────────────────────────────────────

def _reset_cache():
    query_engine._cache["stop_words"] = None
    query_engine._cache["suffix_rules"] = None


def _make_cursor(fetchall_value=None, fetchone_value=None):
    c = MagicMock()
    c.fetchall.return_value = fetchall_value or []
    c.fetchone.return_value = fetchone_value
    return c


def _query_conn(stop_words=None, suffix_rules=None, term_rows=None,
                term_cursor_for_build=None, sim_rows=None, doc_rows=None):
    """Build a mock conn for query() that goes through the query_vector path.

    Call order for query() with method='cosine' (query_vector path):
      1. load_stop_words      → conn.cursor()           non-dict, tuples
      2. load_suffix_rules    → conn.cursor()           non-dict, tuples
      3. _lookup_term_ids     → conn.cursor(dict=True)  dict rows
      4. _build_query_temp_table → conn.cursor(dict=True)  dict rows (TERM SELECT)
      5. cosine SQL           → conn.cursor(dict=True)  dict rows
      6. _enrich_with_titles  → conn.cursor(dict=True)  dict rows
    """
    sw = _make_cursor(fetchall_value=stop_words or [])
    sr = _make_cursor(fetchall_value=suffix_rules or [])
    tm = _make_cursor(fetchall_value=term_rows or [{"id": 10, "name": "trabajo"}])
    bt = _make_cursor(fetchall_value=term_cursor_for_build or [{"id": 10}])
    sim = _make_cursor(fetchall_value=sim_rows or [{"document_id": 1, "score": 0.9}])
    doc = _make_cursor(fetchall_value=doc_rows or [{"id": 1, "title": "Doc One"}])

    conn = MagicMock()
    conn.cursor.side_effect = [sw, sr, tm, bt, sim, doc]
    return conn


# ── ValueError cases ─────────────────────────────────────────────────────────

def test_query_empty_after_preprocessing():
    """A query of only stop words produces no tokens → ValueError."""
    _reset_cache()
    conn = MagicMock()
    # load_stop_words returns the same words as the query
    sw = _make_cursor(fetchall_value=[("de",), ("la",), ("el",), ("en",), ("y",)])
    sr = _make_cursor(fetchall_value=[])
    conn.cursor.side_effect = [sw, sr]

    with pytest.raises(ValueError, match="no terms after preprocessing"):
        query_engine.query("de la el en y", conn)


def test_query_no_known_terms():
    """Real words absent from TERM table raise ValueError."""
    _reset_cache()
    conn = MagicMock()
    sw = _make_cursor(fetchall_value=[])
    sr = _make_cursor(fetchall_value=[])
    tm = _make_cursor(fetchall_value=[])  # no matches in TERM
    conn.cursor.side_effect = [sw, sr, tm]

    with pytest.raises(ValueError, match="None of the query terms"):
        query_engine.query("xyzzy frobozz quux", conn)


def test_query_method_invalid():
    """Unknown method raises ValueError before any DB access."""
    _reset_cache()
    conn = MagicMock()
    with pytest.raises(ValueError, match="Unknown method"):
        query_engine.query("trabajo", conn, method="fuzzy")


# ── Happy path ───────────────────────────────────────────────────────────────

def test_query_valid():
    """A query with known terms returns list of dicts with required keys."""
    _reset_cache()
    conn = _query_conn()
    results = query_engine.query("trabajo", conn, method="cosine")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "document_id" in r
        assert "title" in r
        assert "score" in r


def test_query_top_n_respected():
    """top_n is forwarded to the similarity function."""
    _reset_cache()
    mock_cosine = MagicMock(return_value=[
        {"document_id": 1, "title": "A", "score": 0.9},
        {"document_id": 2, "title": "B", "score": 0.7},
    ])
    # _METHODS holds direct function references; patch.dict replaces the entry
    with patch.dict(query_engine._METHODS, {"cosine": mock_cosine}):
        conn = MagicMock()
        sw = _make_cursor(fetchall_value=[])
        sr = _make_cursor(fetchall_value=[])
        tm = _make_cursor(fetchall_value=[{"id": 10, "name": "huelg"}])
        conn.cursor.side_effect = [sw, sr, tm]

        results = query_engine.query("huelg", conn, method="cosine", top_n=2)
        assert mock_cosine.call_args[1]["top_n"] == 2
        assert len(results) == 2


# ── compare_documents ────────────────────────────────────────────────────────

def _compare_conn(sim_result):
    """Mock for compare_documents: cosine is patched, only DOCUMENT query remains."""
    conn = MagicMock()
    doc_c = _make_cursor(fetchall_value=[
        {"id": 1, "title": "A"},
        {"id": 2, "title": "B"},
    ])
    conn.cursor.return_value = doc_c
    return conn


def test_compare_documents_returns_interpretation():
    """compare_documents always returns 'interpretation' key."""
    mock_c = MagicMock(return_value=[{"document_id": 2, "title": "B", "score": 0.8}])
    with patch.dict(query_engine._METHODS, {"cosine": mock_c}):
        result = query_engine.compare_documents(1, 2, _compare_conn(0.8))
    assert "interpretation" in result
    assert result["interpretation"] in ("similar", "dissimilar")


def test_compare_documents_high_score_is_similar():
    mock_c = MagicMock(return_value=[{"document_id": 2, "title": "B", "score": 0.9}])
    with patch.dict(query_engine._METHODS, {"cosine": mock_c}):
        result = query_engine.compare_documents(1, 2, _compare_conn(0.9))
    assert result["interpretation"] == "similar"


def test_compare_documents_low_score_is_dissimilar():
    mock_c = MagicMock(return_value=[{"document_id": 2, "title": "B", "score": 0.1}])
    with patch.dict(query_engine._METHODS, {"cosine": mock_c}):
        result = query_engine.compare_documents(1, 2, _compare_conn(0.1))
    assert result["interpretation"] == "dissimilar"


# ── all_similarities ─────────────────────────────────────────────────────────

def test_all_similarities_has_all_methods():
    """all_similarities result has keys for all five methods plus 'document'."""
    with patch("src.query_engine.cosine_similarity", return_value=[]), \
         patch("src.query_engine.dice_similarity", return_value=[]), \
         patch("src.query_engine.jaccard_similarity", return_value=[]), \
         patch("src.query_engine.euclidean_distance", return_value=[]), \
         patch("src.query_engine.manhattan_distance", return_value=[]):

        conn = MagicMock()
        doc_c = _make_cursor(fetchone_value={"id": 1, "title": "Doc A"})
        conn.cursor.return_value = doc_c

        result = query_engine.all_similarities(1, conn, top_n=3)

    assert "document" in result
    for m in ("cosine", "dice", "jaccard", "euclidean", "manhattan"):
        assert m in result, f"Missing key: {m}"


# ── Module-level cache ───────────────────────────────────────────────────────

def test_cache_populated_after_first_call():
    """After one query() call, _cache['stop_words'] is not None."""
    _reset_cache()
    assert query_engine._cache["stop_words"] is None

    conn = _query_conn()
    query_engine.query("trabajo", conn)

    assert query_engine._cache["stop_words"] is not None


def test_cache_not_reloaded_on_second_call():
    """DB is queried for stop words/suffixes only once across two calls."""
    _reset_cache()
    mock_cos = MagicMock(return_value=[{"document_id": 1, "title": "T", "score": 0.5}])

    with patch("src.query_engine.load_stop_words") as mock_sw, \
         patch("src.query_engine.load_suffix_rules") as mock_sr, \
         patch.dict(query_engine._METHODS, {"cosine": mock_cos}):

        mock_sw.return_value = set()
        mock_sr.return_value = []

        # After cache is warm, only _lookup_term_ids cursor is needed per call
        tm1 = _make_cursor(fetchall_value=[{"id": 5, "name": "trabajo"}])
        tm2 = _make_cursor(fetchall_value=[{"id": 5, "name": "trabajo"}])
        conn = MagicMock()
        # First call: load_stop_words + load_suffix_rules are patched (no cursor),
        # only _lookup_term_ids uses conn.cursor
        conn.cursor.side_effect = [tm1, tm2]

        query_engine.query("trabajo", conn)
        query_engine.query("trabajo", conn)

    assert mock_sw.call_count == 1
    assert mock_sr.call_count == 1
