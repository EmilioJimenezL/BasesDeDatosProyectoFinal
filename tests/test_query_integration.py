"""Integration tests for Phase 4 — require a live MySQL docbase connection.

Tests are automatically skipped when the database is unreachable.
"""

import itertools
import os
import pytest

try:
    import mysql.connector
    from dotenv import load_dotenv

    load_dotenv()
    _conn_test = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "docbase"),
    )
    _conn_test.close()
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DB_AVAILABLE, reason="MySQL docbase not reachable")


@pytest.fixture(autouse=True, scope="module")
def reset_query_engine_cache():
    """Reset the module-level cache before integration tests run.

    Unit tests (test_query_engine.py) leave _cache populated with mock data.
    Without this reset, _load_cache() skips reloading and the integration
    tests operate with empty stop-words/suffix-rules, breaking preprocessing.
    """
    from src import query_engine
    query_engine._cache["stop_words"] = None
    query_engine._cache["suffix_rules"] = None


@pytest.fixture(scope="module")
def conn():
    from src.db import get_connection
    c = get_connection()
    yield c
    c.close()


# ── Basic query results ──────────────────────────────────────────────────────

def test_cosine_returns_results(conn):
    from src.query_engine import query
    results = query("contrato de trabajo", conn, method="cosine", top_n=3)
    assert len(results) == 3
    for r in results:
        assert "document_id" in r
        assert "title" in r
        assert "score" in r


def test_all_methods_return_results(conn):
    from src.query_engine import query
    for method in ("cosine", "dice", "jaccard", "euclidean", "manhattan"):
        results = query("jornada laboral", conn, method=method, top_n=3)
        assert len(results) > 0, f"Method '{method}' returned no results"


# ── Known query ranking ──────────────────────────────────────────────────────

def test_known_query_ranking(conn):
    """A labor-law query should surface core labor-rights documents in top 3."""
    from src.query_engine import query
    results = query("jornada laboral salario minimo", conn, method="cosine", top_n=3)
    assert len(results) > 0
    top_titles = [r["title"].lower() for r in results]
    # Top 3 should include at least one core labor-law document (LFT, LSS, Artículo 123, etc.)
    labor_keywords = ("lft", "lss", "lsar", "lftse", "artículo", "lifnvt")
    assert any(any(kw in t for kw in labor_keywords) for t in top_titles), (
        f"Expected a labor-law document in top 3, got: {top_titles}"
    )


# ── Score bounds ─────────────────────────────────────────────────────────────

def test_distance_scores_non_negative(conn):
    from src.query_engine import query
    for method in ("euclidean", "manhattan"):
        for r in query("contrato salario", conn, method=method, top_n=10):
            assert r["score"] >= 0.0, (
                f"{method} returned negative score {r['score']} for doc {r['document_id']}"
            )


def test_similarity_scores_bounded(conn):
    """All cosine/dice/jaccard scores must be in [0, 1].

    IDF is clamped at 0 in matrix_builder.apply_tfidf, so HAS.frequency is
    always non-negative and Cauchy-Schwarz guarantees scores in [0, 1].
    """
    from src.query_engine import query
    for method in ("cosine", "dice", "jaccard"):
        for r in query("contrato salario", conn, method=method, top_n=10):
            assert 0.0 <= r["score"] <= 1.0, (
                f"{method} score {r['score']} out of [0,1] for doc {r['document_id']}"
            )


# ── All document pairs ───────────────────────────────────────────────────────

def test_compare_all_document_pairs(conn):
    """Every unique pair must complete without exceptions; cosine in [0,1]."""
    from src.query_engine import compare_documents, get_all_documents

    docs = get_all_documents(conn)
    ids = [d["id"] for d in docs]
    pairs = list(itertools.combinations(ids, 2))

    best_pair = None
    best_score = -1.0
    worst_pair = None
    worst_score = 2.0

    for a, b in pairs:
        result = compare_documents(a, b, conn, method="cosine")
        score = result["score"]
        assert 0.0 <= score <= 1.0, (
            f"Cosine score {score} out of [0,1] for docs ({a},{b})"
        )
        if score > best_score:
            best_score = score
            best_pair = (result["doc_a"]["title"], result["doc_b"]["title"])
        if score < worst_score:
            worst_score = score
            worst_pair = (result["doc_a"]["title"], result["doc_b"]["title"])

    print(f"\nMost similar  ({best_score:.4f}): {best_pair[0]}  ↔  {best_pair[1]}")
    print(f"Least similar ({worst_score:.4f}): {worst_pair[0]}  ↔  {worst_pair[1]}")


# ── SQL views ────────────────────────────────────────────────────────────────

def test_views_queryable(conn):
    """The three views must exist and return at least one row."""
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM v_cosine_similarity LIMIT 5")
    rows = cursor.fetchall()
    assert len(rows) >= 1
    assert "doc_a_title" in rows[0]
    assert "doc_b_title" in rows[0]
    assert "cosine_score" in rows[0]

    cursor.execute("SELECT * FROM v_euclidean_distance LIMIT 5")
    rows = cursor.fetchall()
    assert len(rows) >= 1

    cursor.execute("SELECT * FROM v_document_norms LIMIT 10")
    rows = cursor.fetchall()
    assert len(rows) >= 1

    cursor.close()
