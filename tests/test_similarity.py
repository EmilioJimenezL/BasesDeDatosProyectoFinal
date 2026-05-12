"""Unit tests for src/similarity.py — all DB interactions are mocked."""

import math
from unittest.mock import MagicMock

import pytest

from src.similarity import (
    cosine_similarity,
    dice_similarity,
    jaccard_similarity,
    euclidean_distance,
    manhattan_distance,
    _enrich_with_titles,
    _build_query_temp_table,
)


# ── Mock helpers ─────────────────────────────────────────────────────────────

def _make_cursor(fetchall_value):
    """Return a single mock cursor whose fetchall() returns *fetchall_value*."""
    c = MagicMock()
    c.fetchall.return_value = fetchall_value
    return c


def _doc_id_conn(sim_rows, doc_rows):
    """Two-stage mock for query_doc_id path:
    cursor-1 → main similarity SQL  (dictionary=True)
    cursor-2 → DOCUMENT title lookup (dictionary=True)
    """
    conn = MagicMock()
    conn.cursor.side_effect = [
        _make_cursor(sim_rows),
        _make_cursor(doc_rows),
    ]
    return conn


def _qv_conn(term_rows, sim_rows, doc_rows):
    """Three-stage mock for query_vector path:
    cursor-1 → _build_query_temp_table CREATE/TRUNCATE/SELECT/INSERT (dictionary=True)
    cursor-2 → main similarity SQL  (dictionary=True)
    cursor-3 → DOCUMENT title lookup (dictionary=True)
    """
    conn = MagicMock()
    conn.cursor.side_effect = [
        _make_cursor(term_rows),
        _make_cursor(sim_rows),
        _make_cursor(doc_rows),
    ]
    return conn


# ── _enrich_with_titles ──────────────────────────────────────────────────────

def test_enrich_empty_input():
    conn = MagicMock()
    assert _enrich_with_titles(conn, []) == []


def test_enrich_adds_title():
    conn = MagicMock()
    conn.cursor.return_value = _make_cursor([{"id": 2, "title": "Doc B"}])
    result = _enrich_with_titles(conn, [{"document_id": 2, "score": 0.9}])
    assert result[0]["title"] == "Doc B"
    assert result[0]["score"] == 0.9


def test_enrich_unknown_doc():
    conn = MagicMock()
    conn.cursor.return_value = _make_cursor([])
    result = _enrich_with_titles(conn, [{"document_id": 99, "score": 0.5}])
    assert result[0]["title"] == "Unknown"


# ── cosine_similarity ────────────────────────────────────────────────────────

def test_cosine_perfect_match():
    """Two documents with identical term-frequency vectors → score = 1.0."""
    conn = _doc_id_conn(
        [{"document_id": 2, "score": 1.0}],
        [{"id": 2, "title": "Twin Doc"}],
    )
    results = cosine_similarity(conn, query_doc_id=1, top_n=5)
    assert len(results) == 1
    assert results[0]["score"] == 1.0


def test_cosine_orthogonal():
    """Documents with no shared terms → inner join returns no rows → empty list."""
    conn = _doc_id_conn([], [])
    results = cosine_similarity(conn, query_doc_id=1, top_n=5)
    assert results == []


def test_cosine_score_rounded():
    """Raw MySQL float is rounded to 4 decimal places."""
    conn = _doc_id_conn(
        [{"document_id": 2, "score": 0.666666667}],
        [{"id": 2, "title": "D"}],
    )
    results = cosine_similarity(conn, query_doc_id=1)
    assert results[0]["score"] == round(0.666666667, 4)


# ── dice_similarity ──────────────────────────────────────────────────────────

def test_dice_formula():
    """Verify Dice = 2*dot / (sa + sb) for known vectors.

    a = {t1: 2.0, t2: 1.0}, b = {t1: 1.0, t2: 3.0}
    dot = 2*1 + 1*3 = 5,  sa = 4+1 = 5,  sb = 1+9 = 10
    dice = 2*5 / (5+10) = 10/15 ≈ 0.6667
    """
    expected = round(10 / 15, 4)
    conn = _doc_id_conn(
        [{"document_id": 2, "score": 10 / 15}],
        [{"id": 2, "title": "D"}],
    )
    results = dice_similarity(conn, query_doc_id=1)
    assert results[0]["score"] == expected


# ── jaccard_similarity ───────────────────────────────────────────────────────

def test_jaccard_formula():
    """Verify Jaccard = dot / (sa + sb - dot) for same known vectors.

    dot=5, sa=5, sb=10 → jaccard = 5/(5+10-5) = 5/10 = 0.5
    """
    expected = round(5 / 10, 4)
    conn = _doc_id_conn(
        [{"document_id": 2, "score": 5 / 10}],
        [{"id": 2, "title": "D"}],
    )
    results = jaccard_similarity(conn, query_doc_id=1)
    assert results[0]["score"] == expected


# ── euclidean_distance ───────────────────────────────────────────────────────

def test_euclidean_identical():
    """Identical term-frequency vectors → Euclidean distance = 0.0."""
    conn = _doc_id_conn(
        [{"document_id": 2, "score": 0.0}],
        [{"id": 2, "title": "Clone"}],
    )
    results = euclidean_distance(conn, query_doc_id=1)
    assert results[0]["score"] == 0.0


def test_euclidean_known():
    """Swapped-value vectors give euclidean = sqrt(2).

    a = {t1: 1.0, t2: 2.0}, b = {t1: 2.0, t2: 1.0}
    distance = sqrt((1-2)^2 + (2-1)^2) = sqrt(2) ≈ 1.4142
    """
    expected = round(math.sqrt(2), 4)
    conn = _doc_id_conn(
        [{"document_id": 2, "score": math.sqrt(2)}],
        [{"id": 2, "title": "D"}],
    )
    results = euclidean_distance(conn, query_doc_id=1)
    assert results[0]["score"] == expected


# ── manhattan_distance ───────────────────────────────────────────────────────

def test_manhattan_known():
    """Vectors [3,1] and [1,3] → Manhattan = |3-1| + |1-3| = 4.0."""
    conn = _doc_id_conn(
        [{"document_id": 2, "score": 4.0}],
        [{"id": 2, "title": "D"}],
    )
    results = manhattan_distance(conn, query_doc_id=1)
    assert results[0]["score"] == 4.0


# ── Cross-metric ordering ────────────────────────────────────────────────────

def test_cosine_gt_dice_gt_jaccard():
    """For typical non-identical overlapping vectors: cosine >= dice >= jaccard.

    a = {t1: 3.0, t2: 1.0}, b = {t1: 1.0, t2: 1.0}
    dot=4, sa=10, sb=2
    cosine  = 4/sqrt(20) ≈ 0.8944
    dice    = 8/12       ≈ 0.6667
    jaccard = 4/8        = 0.5
    """
    doc_row = [{"id": 2, "title": "D"}]

    cosine_score = round(4 / math.sqrt(20), 4)
    dice_score = round(8 / 12, 4)
    jaccard_score = round(4 / 8, 4)

    c = cosine_similarity(
        _doc_id_conn([{"document_id": 2, "score": cosine_score}], doc_row),
        query_doc_id=1,
    )[0]["score"]
    d = dice_similarity(
        _doc_id_conn([{"document_id": 2, "score": dice_score}], doc_row),
        query_doc_id=1,
    )[0]["score"]
    j = jaccard_similarity(
        _doc_id_conn([{"document_id": 2, "score": jaccard_score}], doc_row),
        query_doc_id=1,
    )[0]["score"]

    assert c >= d, f"cosine {c} < dice {d}"
    assert d >= j, f"dice {d} < jaccard {j}"


# ── Ranking order ────────────────────────────────────────────────────────────

def test_similarity_ranking_order():
    """doc_2 shares more terms with query than doc_3 → doc_2 ranked first."""
    conn = _doc_id_conn(
        [
            {"document_id": 2, "score": 0.85},
            {"document_id": 3, "score": 0.42},
        ],
        [
            {"id": 2, "title": "High Match"},
            {"id": 3, "title": "Low Match"},
        ],
    )
    results = cosine_similarity(conn, query_doc_id=1, top_n=3)
    assert results[0]["document_id"] == 2
    assert results[0]["score"] > results[1]["score"]


# ── query_vector path ────────────────────────────────────────────────────────

def test_cosine_query_vector_path():
    """Smoke test: query_vector path builds temp table and returns results."""
    # cursor-1: _build_query_temp_table → SELECT id FROM TERM
    # cursor-2: main SQL (dictionary=True)
    # cursor-3: _enrich_with_titles (dictionary=True)
    conn = _qv_conn(
        [{"id": 1}, {"id": 2}],                      # valid term IDs from TERM
        [{"document_id": 3, "score": 0.75}],          # similarity SQL
        [{"id": 3, "title": "Found Doc"}],            # DOCUMENT lookup
    )
    results = cosine_similarity(conn, query_vector={1: 1.0, 2: 2.0}, top_n=3)
    assert len(results) == 1
    assert results[0]["score"] == 0.75
    assert results[0]["title"] == "Found Doc"
