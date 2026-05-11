"""Integration test: full LSI pipeline with a tiny in-memory corpus.

No real PDFs and no MySQL connection required.
"""

import numpy as np
import pytest

from src.preprocessor import preprocess
from src.matrix_builder import build_frequency_matrix, apply_tfidf
from src.svd_engine import compute_svd, project_query


# ── Helpers ──────────────────────────────────────────────────────────────────

MOCK_STOP_WORDS = {"de", "la", "el", "en", "y", "a", "que", "se", "los", "las",
                   "un", "una", "por", "con", "del", "es", "para", "como"}

MOCK_SUFFIX_RULES = [
    ("ciones", ""),
    ("ción", ""),
    ("mente", ""),
    ("ando", ""),
    ("ado", ""),
    ("ido", ""),
    ("ista", ""),
    ("ismo", ""),
    ("idad", ""),
    ("ble", ""),
    ("dor", ""),
    ("nte", ""),
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Fake corpus ──────────────────────────────────────────────────────────────

FAKE_DOCS = {
    "doc1": (
        "El contrato de trabajo establece las condiciones laborales del trabajador. "
        "El patrón firma el contrato laboral con el trabajador asalariado."
    ),
    "doc2": (
        "La jornada laboral no puede exceder de ocho horas diarias. "
        "Las horas extras deben compensarse con un salario adicional."
    ),
    "doc3": (
        "El sindicato convoca a huelga por mejores condiciones salariales. "
        "La huelga fue declarada legal por el tribunal laboral."
    ),
    "doc4": (
        "Las prestaciones incluyen vacaciones, aguinaldo y seguro social. "
        "El trabajador tiene derecho a prestaciones desde el primer día laboral."
    ),
    "doc5": (
        "La capacitación y adiestramiento son obligaciones del patrón. "
        "El programa de capacitación debe ser aprobado por la comisión mixta."
    ),
}


# ── Integration test ─────────────────────────────────────────────────────────

def test_full_pipeline_e2e():
    # 1. Preprocess fake documents
    processed = {}
    for name, text in FAKE_DOCS.items():
        tokens = preprocess(text, MOCK_STOP_WORDS, MOCK_SUFFIX_RULES)
        processed[name] = tokens
        assert len(tokens) > 0, f"{name} produced no tokens"

    # 2. Build frequency matrix
    term_index, doc_index, matrix = build_frequency_matrix(processed)
    assert len(term_index) > 5
    assert len(doc_index) == 5

    # 3. Apply TF-IDF
    weighted = apply_tfidf(matrix, term_index, doc_index)
    assert len(weighted) == len(term_index)
    assert len(weighted[0]) == len(doc_index)

    # 4. Compute SVD with k=2
    U, s, Vt = compute_svd(weighted, k=2)
    assert U.shape[1] == 2
    assert s.shape == (2,)
    assert Vt.shape[0] == 2

    # 5. Project a query about "contrato trabajo"
    query = "contrato de trabajo laboral"
    query_tokens = preprocess(query, MOCK_STOP_WORDS, MOCK_SUFFIX_RULES)
    q_vec = project_query(query_tokens, term_index, U, s)

    # 6. Assert shape
    assert q_vec.shape == (2,)

    # 7. Query about "contrato trabajo" should be closer to doc1
    #    (which is about contracts) than to doc5 (about training)
    doc1_vec = Vt[:, doc_index["doc1"]] * s
    doc5_vec = Vt[:, doc_index["doc5"]] * s

    sim_doc1 = _cosine(q_vec, doc1_vec)
    sim_doc5 = _cosine(q_vec, doc5_vec)

    assert sim_doc1 > sim_doc5, (
        f"Query 'contrato de trabajo' should be more similar to doc1 "
        f"(contracts) than doc5 (training): sim1={sim_doc1:.4f} vs sim5={sim_doc5:.4f}"
    )
