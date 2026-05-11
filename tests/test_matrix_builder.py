"""Unit tests for src/matrix_builder.py."""

import pytest
from src.matrix_builder import build_frequency_matrix, apply_tfidf


# ── Test data ────────────────────────────────────────────────────────────────

def _make_docs():
    """Three small fake documents for testing."""
    return {
        "doc_a": ["trabajo", "salario", "contrato", "trabajo"],
        "doc_b": ["salario", "jornada", "salario"],
        "doc_c": ["contrato", "jornada", "trabajo", "salario", "huelga"],
    }


# ── build_frequency_matrix tests ────────────────────────────────────────────

def test_build_frequency_matrix_shape():
    docs = _make_docs()
    term_index, doc_index, matrix = build_frequency_matrix(docs)

    # Unique terms: contrato, huelga, jornada, salario, trabajo → 5
    assert len(term_index) == 5
    assert len(doc_index) == 3
    assert len(matrix) == 5       # rows = terms
    assert len(matrix[0]) == 3    # cols = docs


def test_build_frequency_matrix_counts():
    docs = _make_docs()
    term_index, doc_index, matrix = build_frequency_matrix(docs)

    # doc_a has "trabajo" twice
    ti = term_index["trabajo"]
    di = doc_index["doc_a"]
    assert matrix[ti][di] == 2.0

    # doc_b has "salario" twice
    ti = term_index["salario"]
    di = doc_index["doc_b"]
    assert matrix[ti][di] == 2.0

    # doc_c has "huelga" once
    ti = term_index["huelga"]
    di = doc_index["doc_c"]
    assert matrix[ti][di] == 1.0

    # doc_a has no "huelga"
    ti = term_index["huelga"]
    di = doc_index["doc_a"]
    assert matrix[ti][di] == 0.0


def test_build_frequency_matrix_empty_doc():
    docs = {
        "doc_a": ["trabajo", "salario"],
        "doc_b": [],  # empty doc
        "doc_c": ["trabajo"],
    }
    term_index, doc_index, matrix = build_frequency_matrix(docs)
    # Should not crash; doc_b column should be all zeros
    di = doc_index["doc_b"]
    for row in matrix:
        assert row[di] == 0.0


# ── apply_tfidf tests ───────────────────────────────────────────────────────

def test_apply_tfidf_reduces_common_terms():
    docs = _make_docs()
    term_index, doc_index, matrix = build_frequency_matrix(docs)
    weighted = apply_tfidf(matrix, term_index, doc_index)

    # "salario" appears in all 3 docs → high df → low IDF → lower weight
    # "huelga" appears in only 1 doc → low df → high IDF → higher weight
    ti_salario = term_index["salario"]
    ti_huelga = term_index["huelga"]
    di_c = doc_index["doc_c"]

    # Both appear once in doc_c, so TF is same; IDF should differentiate
    assert weighted[ti_huelga][di_c] > weighted[ti_salario][di_c]


def test_apply_tfidf_output_shape():
    docs = _make_docs()
    term_index, doc_index, matrix = build_frequency_matrix(docs)
    weighted = apply_tfidf(matrix, term_index, doc_index)

    assert len(weighted) == len(matrix)
    assert len(weighted[0]) == len(matrix[0])
