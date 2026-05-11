"""End-to-end query processing pipeline.

Accepts a raw query string, preprocesses it, builds a term-frequency vector,
and delegates similarity scoring to MySQL via the functions in similarity.py.
"""

from src.preprocessor import preprocess_query, load_stop_words, load_suffix_rules
from src.similarity import (
    cosine_similarity,
    dice_similarity,
    jaccard_similarity,
    euclidean_distance,
    manhattan_distance,
)

_METHODS = {
    "cosine": cosine_similarity,
    "dice": dice_similarity,
    "jaccard": jaccard_similarity,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}

_cache: dict = {"stop_words": None, "suffix_rules": None}


def _load_cache(conn) -> None:
    if _cache["stop_words"] is None:
        _cache["stop_words"] = load_stop_words(conn)
    if _cache["suffix_rules"] is None:
        _cache["suffix_rules"] = load_suffix_rules(conn)


def _lookup_term_ids(conn, tokens: list) -> dict:
    """Return {term_id: frequency} for tokens that exist in the TERM table."""
    if not tokens:
        return {}
    unique = list(set(tokens))
    fmt = ",".join(["%s"] * len(unique))
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"SELECT id, name FROM TERM WHERE name IN ({fmt})", unique)
    name_to_id = {row["name"]: row["id"] for row in cursor.fetchall()}
    cursor.close()
    freq: dict = {}
    for token in tokens:
        tid = name_to_id.get(token)
        if tid is not None:
            freq[tid] = freq.get(tid, 0) + 1
    return freq


def query(text: str, conn, method: str = "cosine",
          top_n: int = 10, k: int = 9) -> list:
    """Preprocess *text*, build a term-frequency vector, and rank documents.

    Parameters
    ----------
    text:   Raw query string.
    conn:   Active MySQL connection.
    method: One of 'cosine', 'dice', 'jaccard', 'euclidean', 'manhattan'.
    top_n:  Maximum number of results to return.
    k:      SVD rank (reserved for future LSI projection; unused in Phase 4).

    Raises
    ------
    ValueError: if preprocessing produces no tokens or none exist in the corpus.
    ValueError: if *method* is not one of the five supported values.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid options: {', '.join(_METHODS)}"
        )

    _load_cache(conn)
    tokens = preprocess_query(text, _cache["stop_words"], _cache["suffix_rules"])

    if not tokens:
        raise ValueError(
            "Query produced no terms after preprocessing. Try different words."
        )

    query_vector = _lookup_term_ids(conn, tokens)

    if not query_vector:
        raise ValueError("None of the query terms exist in the corpus.")

    return _METHODS[method](conn, query_vector=query_vector, top_n=top_n)


def compare_documents(doc_id_a: int, doc_id_b: int, conn,
                      method: str = "cosine") -> dict:
    """Compute the pairwise similarity/distance between two corpus documents.

    Returns a dict with keys: doc_a, doc_b, method, score, interpretation.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'.")

    results = _METHODS[method](conn, query_doc_id=doc_id_a, top_n=9999)
    score = 0.0
    for r in results:
        if r["document_id"] == doc_id_b:
            score = r["score"]
            break

    is_distance = method in ("euclidean", "manhattan")
    if is_distance:
        interpretation = "similar" if score < 1.0 else "dissimilar"
    else:
        interpretation = "similar" if score > 0.5 else "dissimilar"

    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, title FROM DOCUMENT WHERE id IN (%s, %s)",
        (doc_id_a, doc_id_b),
    )
    titles = {row["id"]: row["title"] for row in cursor.fetchall()}
    cursor.close()

    return {
        "doc_a": {"id": doc_id_a, "title": titles.get(doc_id_a, "Unknown")},
        "doc_b": {"id": doc_id_b, "title": titles.get(doc_id_b, "Unknown")},
        "method": method,
        "score": score,
        "interpretation": interpretation,
    }


def all_similarities(doc_id: int, conn, top_n: int = 5) -> dict:
    """Run all five metrics against *doc_id* and return results side by side."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title FROM DOCUMENT WHERE id = %s", (doc_id,))
    row = cursor.fetchone()
    cursor.close()
    title = row["title"] if row else "Unknown"

    return {
        "document": {"id": doc_id, "title": title},
        "cosine":    cosine_similarity(conn, query_doc_id=doc_id, top_n=top_n),
        "dice":      dice_similarity(conn, query_doc_id=doc_id, top_n=top_n),
        "jaccard":   jaccard_similarity(conn, query_doc_id=doc_id, top_n=top_n),
        "euclidean": euclidean_distance(conn, query_doc_id=doc_id, top_n=top_n),
        "manhattan": manhattan_distance(conn, query_doc_id=doc_id, top_n=top_n),
    }


def get_all_documents(conn) -> list:
    """Return all rows from DOCUMENT as a list of dicts."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, url, author, doc_date FROM DOCUMENT ORDER BY id")
    rows = cursor.fetchall()
    cursor.close()
    return rows
