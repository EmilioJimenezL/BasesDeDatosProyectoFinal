"""Term-document matrix construction.

Builds a raw frequency matrix from preprocessed token lists, applies TF-IDF
weighting, and persists terms and frequency data to the TERM, WORD, and HAS
tables in MySQL.
"""

import math


def build_frequency_matrix(
    docs: dict[str, list[str]],
) -> tuple[dict, dict, list[list[float]]]:
    """Build a raw term-frequency matrix from preprocessed documents.

    Parameters
    ----------
    docs : dict mapping doc_name → list of preprocessed tokens.

    Returns
    -------
    term_index : dict mapping term string → row index (0-based).
    doc_index  : dict mapping doc_name → column index (0-based).
    matrix     : 2-D list  [term_idx][doc_idx] = raw frequency count.
    """
    # Collect all unique terms across every document
    all_terms: set[str] = set()
    for tokens in docs.values():
        all_terms.update(tokens)

    term_index = {term: i for i, term in enumerate(sorted(all_terms))}
    doc_names = sorted(docs.keys())
    doc_index = {name: j for j, name in enumerate(doc_names)}

    num_terms = len(term_index)
    num_docs = len(doc_index)

    # Initialise zero matrix
    matrix = [[0.0] * num_docs for _ in range(num_terms)]

    for doc_name, tokens in docs.items():
        j = doc_index[doc_name]
        for token in tokens:
            i = term_index[token]
            matrix[i][j] += 1.0

    return term_index, doc_index, matrix


def apply_tfidf(
    matrix: list[list[float]],
    term_index: dict,
    doc_index: dict,
) -> list[list[float]]:
    """Apply TF-IDF weighting to a raw frequency matrix.

    TF  = frequency / total_terms_in_document
    IDF = ln(N / (1 + df))   where df = number of docs containing the term.
    """
    num_terms = len(term_index)
    num_docs = len(doc_index)

    # Total tokens per document column
    doc_totals = [0.0] * num_docs
    for j in range(num_docs):
        for i in range(num_terms):
            doc_totals[j] += matrix[i][j]

    # Document frequency per term row
    doc_freq = [0] * num_terms
    for i in range(num_terms):
        for j in range(num_docs):
            if matrix[i][j] > 0:
                doc_freq[i] += 1

    # Build weighted matrix
    weighted = [[0.0] * num_docs for _ in range(num_terms)]
    for i in range(num_terms):
        idf = math.log(num_docs / (1 + doc_freq[i]))
        for j in range(num_docs):
            if matrix[i][j] > 0 and doc_totals[j] > 0:
                tf = matrix[i][j] / doc_totals[j]
                weighted[i][j] = tf * idf
    return weighted


def save_terms(conn, term_index: dict) -> dict[str, int]:
    """Insert all terms into TERM and WORD tables.

    Uses INSERT IGNORE on TERM so re-running is safe.
    Returns a dict mapping term string → MySQL term id.
    """
    cursor = conn.cursor()
    term_db_ids: dict[str, int] = {}

    for term in term_index:
        cursor.execute("INSERT IGNORE INTO TERM (name) VALUES (%s)", (term,))
        conn.commit()

    # Fetch the ids back (INSERT IGNORE doesn't guarantee lastrowid)
    cursor.execute("SELECT id, name FROM TERM")
    for row in cursor.fetchall():
        term_db_ids[row[1]] = row[0]

    # Insert word forms into WORD table (same string for now)
    for term, term_id in term_db_ids.items():
        cursor.execute(
            "SELECT id FROM WORD WHERE term_id = %s AND word = %s",
            (term_id, term),
        )
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO WORD (term_id, word) VALUES (%s, %s)",
                (term_id, term),
            )
    conn.commit()

    cursor.close()
    return term_db_ids


def save_matrix(
    conn,
    matrix: list[list[float]],
    term_index: dict,
    doc_index: dict,
    term_db_ids: dict[str, int],
    doc_db_ids: dict[str, int],
) -> int:
    """Insert non-zero matrix cells into HAS in batches of 500.

    Uses ON DUPLICATE KEY UPDATE so re-running is safe.
    Returns total rows inserted/updated.
    """
    cursor = conn.cursor()
    sql = (
        "INSERT INTO HAS (document_id, term_id, frequency) "
        "VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE frequency=VALUES(frequency)"
    )

    batch: list[tuple] = []
    total = 0

    for term, i in term_index.items():
        if term not in term_db_ids:
            continue
        tid = term_db_ids[term]
        for doc_name, j in doc_index.items():
            if matrix[i][j] == 0.0:
                continue
            if doc_name not in doc_db_ids:
                continue
            did = doc_db_ids[doc_name]
            batch.append((did, tid, matrix[i][j]))
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
    return total
