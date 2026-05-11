"""Truncated SVD (LSI) engine.

Applies scipy.sparse.linalg.svds to the term-document matrix, stores the
resulting T, S, D matrices at rank k in the SVD_MATRIX table, and exposes
helpers to load pre-computed decompositions and project queries.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def compute_svd(
    matrix: list[list[float]], k: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute truncated SVD on a terms × documents matrix.

    Parameters
    ----------
    matrix : 2-D list  [terms][docs]  (TF-IDF weighted values).
    k      : desired rank (will be clamped to min(shape) - 1 if too large).

    Returns
    -------
    U  : ndarray (M, k)  — term vectors.
    s  : ndarray (k,)    — singular values (descending order).
    Vt : ndarray (k, N)  — document vectors.
    """
    A = np.array(matrix, dtype=np.float64)

    # Clamp k so svds doesn't crash
    max_k = min(A.shape) - 1
    if max_k < 1:
        max_k = 1
    k = min(k, max_k)

    A_sparse = csr_matrix(A)
    U, s, Vt = svds(A_sparse, k=k)

    # svds returns singular values in ascending order — reverse to descending
    idx = np.argsort(s)[::-1]
    U = U[:, idx]
    s = s[idx]
    Vt = Vt[idx, :]

    return U, s, Vt


def save_svd(
    conn,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    term_index: dict,
    doc_index: dict,
    term_db_ids: dict[str, int],
    doc_db_ids: dict[str, int],
    k: int,
) -> None:
    """Persist SVD decomposition to the SVD_MATRIX table.

    Deletes existing rows for this k first, then batch-inserts all
    (term_id, document_id, t_value, s_value, d_value, k_rank) rows.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM SVD_MATRIX WHERE k_rank = %s", (k,))
    conn.commit()

    sql = (
        "INSERT INTO SVD_MATRIX (term_id, document_id, t_value, s_value, d_value, k_rank) "
        "VALUES (%s, %s, %s, %s, %s, %s)"
    )

    # Sort indices to ensure consistent ordering
    terms_sorted = sorted(term_index.items(), key=lambda x: x[1])
    docs_sorted = sorted(doc_index.items(), key=lambda x: x[1])

    actual_k = len(s)
    batch: list[tuple] = []
    total = 0

    for ki in range(actual_k):
        for term_name, term_idx in terms_sorted:
            if term_name not in term_db_ids:
                continue
            tid = term_db_ids[term_name]
            for doc_name, doc_idx in docs_sorted:
                if doc_name not in doc_db_ids:
                    continue
                did = doc_db_ids[doc_name]
                batch.append((
                    tid,
                    did,
                    float(U[term_idx, ki]),
                    float(s[ki]),
                    float(Vt[ki, doc_idx]),
                    k,
                ))
                if len(batch) >= 500:
                    cursor.executemany(sql, batch)
                    conn.commit()
                    total += len(batch)
                    batch = []

    if batch:
        cursor.executemany(sql, batch)
        conn.commit()
        total += len(batch)

    cursor.close()


def get_document_vectors(conn, k: int) -> dict[int, np.ndarray]:
    """Load document vectors from SVD_MATRIX for a given k.

    Returns dict mapping document_id → numpy vector of length k.
    Vector for doc j = [Vt[i][j] * s[i] for i in range(k)]
    (scaled document vector in LSI space).
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT DISTINCT k_rank FROM SVD_MATRIX WHERE k_rank = %s",
        (k,),
    )
    if not cursor.fetchone():
        cursor.close()
        return {}

    # Get unique documents
    cursor.execute(
        "SELECT DISTINCT document_id FROM SVD_MATRIX WHERE k_rank = %s",
        (k,),
    )
    doc_ids = [row["document_id"] for row in cursor.fetchall()]

    doc_vectors: dict[int, np.ndarray] = {}

    for did in doc_ids:
        cursor.execute(
            """
            SELECT d_value, s_value
            FROM SVD_MATRIX
            WHERE k_rank = %s AND document_id = %s
            GROUP BY d_value, s_value, term_id
            ORDER BY term_id
            LIMIT %s
            """,
            (k, did, k),
        )
        rows = cursor.fetchall()
        # We need exactly one (d_value, s_value) per SVD component
        # Group by unique s_value (each component has a unique singular value)
        # Actually — the SVD_MATRIX stores one row per (term, doc, component).
        # For document vectors we need d_value * s_value per component.
        # We can derive the component vector from distinct s_value groups.
        pass

    # Better approach: query distinct components
    cursor.execute(
        """
        SELECT document_id, d_value, s_value
        FROM SVD_MATRIX
        WHERE k_rank = %s
        ORDER BY document_id, s_value DESC
        """,
        (k,),
    )
    all_rows = cursor.fetchall()

    # Group by document_id — for each doc we want one d_value per component.
    # Since each (term, doc, component) row has the same d_value and s_value
    # for a given (doc, component), we can just take distinct ones.
    from collections import defaultdict

    doc_components: dict[int, list[tuple[float, float]]] = defaultdict(list)
    seen: dict[int, set] = defaultdict(set)

    for row in all_rows:
        did = row["document_id"]
        sv = row["s_value"]
        dv = row["d_value"]
        key = (sv, dv)
        if key not in seen[did]:
            seen[did].add(key)
            doc_components[did].append((sv, dv))

    for did, components in doc_components.items():
        vec = np.array([sv * dv for sv, dv in components])
        doc_vectors[did] = vec

    cursor.close()
    return doc_vectors


def project_query(
    query_tokens: list[str],
    term_index: dict,
    U: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Project preprocessed query tokens into LSI space.

    Builds a query vector q of length M (1.0 for each known term, 0.0 otherwise),
    then projects:  q_lsi = q @ U / s

    Returns a numpy array of length k.
    """
    M = U.shape[0]
    q = np.zeros(M)

    for token in query_tokens:
        if token in term_index:
            q[term_index[token]] = 1.0

    # Pseudo-inverse projection into LSI space
    # Avoid division by zero
    s_safe = np.where(s > 1e-12, s, 1.0)
    q_lsi = (q @ U) / s_safe

    return q_lsi
