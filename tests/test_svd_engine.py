"""Unit tests for src/svd_engine.py."""

import numpy as np
import pytest
from src.svd_engine import compute_svd, project_query


# ── compute_svd tests ───────────────────────────────────────────────────────

def _make_matrix(rows=10, cols=5):
    """Return a random non-zero matrix of given shape."""
    rng = np.random.default_rng(42)
    return (rng.random((rows, cols)) * 10).tolist()


def test_compute_svd_shapes():
    matrix = _make_matrix(10, 5)
    U, s, Vt = compute_svd(matrix, k=3)
    assert U.shape == (10, 3)
    assert s.shape == (3,)
    assert Vt.shape == (3, 5)


def test_compute_svd_k_clamped():
    # 10×5 matrix → max k = min(10,5)-1 = 4
    matrix = _make_matrix(10, 5)
    U, s, Vt = compute_svd(matrix, k=100)
    assert s.shape[0] <= 4  # clamped
    assert U.shape[1] == s.shape[0]
    assert Vt.shape[0] == s.shape[0]


def test_singular_values_descending():
    matrix = _make_matrix(10, 5)
    _, s, _ = compute_svd(matrix, k=3)
    for i in range(len(s) - 1):
        assert s[i] >= s[i + 1], f"s[{i}]={s[i]} < s[{i+1}]={s[i+1]}"


# ── project_query tests ─────────────────────────────────────────────────────

def test_project_query_shape():
    matrix = _make_matrix(10, 5)
    U, s, Vt = compute_svd(matrix, k=3)

    term_index = {f"term_{i}": i for i in range(10)}
    result = project_query(["term_0", "term_5"], term_index, U, s)
    assert result.shape == (3,)


def test_project_query_known_term():
    matrix = _make_matrix(10, 5)
    U, s, Vt = compute_svd(matrix, k=3)
    term_index = {f"term_{i}": i for i in range(10)}

    # Query with a known term → non-zero vector
    vec_known = project_query(["term_0"], term_index, U, s)
    assert np.linalg.norm(vec_known) > 1e-8

    # Query with only unknown terms → zero vector
    vec_unknown = project_query(["unknown_xyz"], term_index, U, s)
    assert np.linalg.norm(vec_unknown) < 1e-8
