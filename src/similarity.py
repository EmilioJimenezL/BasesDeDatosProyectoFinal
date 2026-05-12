"""Similarity and distance functions computed via SQL queries inside MySQL.

All five metrics are evaluated entirely within the database engine — Python
only passes parameters and processes the ranked result rows.
"""


def _build_query_temp_table(conn, query_vector: dict) -> None:
    """Create (if absent) and repopulate QUERY_VECTOR with the given term frequencies.

    Keys of *query_vector* must be TERM.id integers.
    Only terms that exist in the TERM table are inserted (unknown terms ignored).
    TRUNCATE + re-insert on every call so successive queries in the same session
    never see stale data.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS QUERY_VECTOR (
            term_id   INT   NOT NULL,
            frequency FLOAT NOT NULL,
            PRIMARY KEY (term_id)
        )
    """)
    cursor.execute("TRUNCATE TABLE QUERY_VECTOR")
    if query_vector:
        term_ids = list(query_vector.keys())
        fmt = ",".join(["%s"] * len(term_ids))
        cursor.execute(f"SELECT id FROM TERM WHERE id IN ({fmt})", term_ids)
        valid_ids = {row["id"] for row in cursor.fetchall()}
        rows = [
            (int(tid), float(freq))
            for tid, freq in query_vector.items()
            if tid in valid_ids
        ]
        if rows:
            cursor.executemany(
                "INSERT INTO QUERY_VECTOR (term_id, frequency) VALUES (%s, %s)",
                rows,
            )
    cursor.close()


def _enrich_with_titles(conn, rows: list) -> list:
    """Attach *title* from DOCUMENT to each row dict that has a *document_id* key."""
    if not rows:
        return []
    doc_ids = [r["document_id"] for r in rows]
    fmt = ",".join(["%s"] * len(doc_ids))
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        f"SELECT id, title FROM DOCUMENT WHERE id IN ({fmt})",
        doc_ids,
    )
    doc_map = {row["id"]: row["title"] for row in cursor.fetchall()}
    cursor.close()
    return [
        {
            "document_id": r["document_id"],
            "title": doc_map.get(r["document_id"], "Unknown"),
            "score": r["score"],
        }
        for r in rows
    ]


def _run_similarity_sql(conn, sql_doc_id, sql_qv, params_doc_id, params_qv,
                        query_doc_id, query_vector, top_n, order_desc=True):
    """Shared execution helper for all five metric functions.

    Opens the SQL cursor *after* building the temp table so that cursor
    ordering in the MySQL session is predictable.
    """
    if query_doc_id is not None:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_doc_id, params_doc_id + (top_n,))
    else:
        _build_query_temp_table(conn, query_vector or {})
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_qv, params_qv + (top_n,))
    raw = cursor.fetchall()
    cursor.close()
    rows = [
        {
            "document_id": r["document_id"],
            "score": round(float(r["score"]) if r["score"] is not None else 0.0, 4),
        }
        for r in raw
    ]
    return _enrich_with_titles(conn, rows)


# ── Cosine similarity ────────────────────────────────────────────────────────

_SQL_COSINE_DOC = """
    SELECT
        b.document_id,
        SUM(a.frequency * b.frequency)
          / (SQRT(SUM(a.frequency * a.frequency))
             * SQRT(SUM(b.frequency * b.frequency))) AS score
    FROM HAS a
    JOIN HAS b ON a.term_id = b.term_id
    WHERE a.document_id = %s
      AND b.document_id != %s
    GROUP BY b.document_id
    ORDER BY score DESC
    LIMIT %s
"""

_SQL_COSINE_QV = """
    SELECT
        b.document_id,
        SUM(a.frequency * b.frequency)
          / (SQRT(SUM(a.frequency * a.frequency))
             * SQRT(SUM(b.frequency * b.frequency))) AS score
    FROM QUERY_VECTOR a
    JOIN HAS b ON a.term_id = b.term_id
    GROUP BY b.document_id
    ORDER BY score DESC
    LIMIT %s
"""


def cosine_similarity(conn, query_doc_id: int = None,
                      query_vector: dict = None,
                      top_n: int = 10) -> list:
    return _run_similarity_sql(
        conn,
        _SQL_COSINE_DOC, _SQL_COSINE_QV,
        (query_doc_id, query_doc_id), (),
        query_doc_id, query_vector, top_n,
    )


# ── Dice similarity ──────────────────────────────────────────────────────────

_SQL_DICE_DOC = """
    SELECT
        b.document_id,
        2 * SUM(a.frequency * b.frequency)
          / (SUM(a.frequency * a.frequency) + SUM(b.frequency * b.frequency)) AS score
    FROM HAS a
    JOIN HAS b ON a.term_id = b.term_id
    WHERE a.document_id = %s
      AND b.document_id != %s
    GROUP BY b.document_id
    ORDER BY score DESC
    LIMIT %s
"""

_SQL_DICE_QV = """
    SELECT
        b.document_id,
        2 * SUM(a.frequency * b.frequency)
          / (SUM(a.frequency * a.frequency) + SUM(b.frequency * b.frequency)) AS score
    FROM QUERY_VECTOR a
    JOIN HAS b ON a.term_id = b.term_id
    GROUP BY b.document_id
    ORDER BY score DESC
    LIMIT %s
"""


def dice_similarity(conn, query_doc_id: int = None,
                    query_vector: dict = None,
                    top_n: int = 10) -> list:
    return _run_similarity_sql(
        conn,
        _SQL_DICE_DOC, _SQL_DICE_QV,
        (query_doc_id, query_doc_id), (),
        query_doc_id, query_vector, top_n,
    )


# ── Jaccard similarity ───────────────────────────────────────────────────────

_SQL_JACCARD_DOC = """
    SELECT
        b.document_id,
        SUM(a.frequency * b.frequency)
          / (SUM(a.frequency * a.frequency)
             + SUM(b.frequency * b.frequency)
             - SUM(a.frequency * b.frequency)) AS score
    FROM HAS a
    JOIN HAS b ON a.term_id = b.term_id
    WHERE a.document_id = %s
      AND b.document_id != %s
    GROUP BY b.document_id
    ORDER BY score DESC
    LIMIT %s
"""

_SQL_JACCARD_QV = """
    SELECT
        b.document_id,
        SUM(a.frequency * b.frequency)
          / (SUM(a.frequency * a.frequency)
             + SUM(b.frequency * b.frequency)
             - SUM(a.frequency * b.frequency)) AS score
    FROM QUERY_VECTOR a
    JOIN HAS b ON a.term_id = b.term_id
    GROUP BY b.document_id
    ORDER BY score DESC
    LIMIT %s
"""


def jaccard_similarity(conn, query_doc_id: int = None,
                       query_vector: dict = None,
                       top_n: int = 10) -> list:
    return _run_similarity_sql(
        conn,
        _SQL_JACCARD_DOC, _SQL_JACCARD_QV,
        (query_doc_id, query_doc_id), (),
        query_doc_id, query_vector, top_n,
    )


# ── Euclidean distance ───────────────────────────────────────────────────────

_SQL_EUCLIDEAN_DOC = """
    SELECT
        b.document_id,
        SQRT(SUM(POW(a.frequency - b.frequency, 2))) AS score
    FROM HAS a
    JOIN HAS b ON a.term_id = b.term_id
    WHERE a.document_id = %s
      AND b.document_id != %s
    GROUP BY b.document_id
    ORDER BY score ASC
    LIMIT %s
"""

_SQL_EUCLIDEAN_QV = """
    SELECT
        b.document_id,
        SQRT(SUM(POW(a.frequency - b.frequency, 2))) AS score
    FROM QUERY_VECTOR a
    JOIN HAS b ON a.term_id = b.term_id
    GROUP BY b.document_id
    ORDER BY score ASC
    LIMIT %s
"""


def euclidean_distance(conn, query_doc_id: int = None,
                       query_vector: dict = None,
                       top_n: int = 10) -> list:
    return _run_similarity_sql(
        conn,
        _SQL_EUCLIDEAN_DOC, _SQL_EUCLIDEAN_QV,
        (query_doc_id, query_doc_id), (),
        query_doc_id, query_vector, top_n,
        order_desc=False,
    )


# ── Manhattan distance ───────────────────────────────────────────────────────

_SQL_MANHATTAN_DOC = """
    SELECT
        b.document_id,
        SUM(ABS(a.frequency - b.frequency)) AS score
    FROM HAS a
    JOIN HAS b ON a.term_id = b.term_id
    WHERE a.document_id = %s
      AND b.document_id != %s
    GROUP BY b.document_id
    ORDER BY score ASC
    LIMIT %s
"""

_SQL_MANHATTAN_QV = """
    SELECT
        b.document_id,
        SUM(ABS(a.frequency - b.frequency)) AS score
    FROM QUERY_VECTOR a
    JOIN HAS b ON a.term_id = b.term_id
    GROUP BY b.document_id
    ORDER BY score ASC
    LIMIT %s
"""


def manhattan_distance(conn, query_doc_id: int = None,
                       query_vector: dict = None,
                       top_n: int = 10) -> list:
    return _run_similarity_sql(
        conn,
        _SQL_MANHATTAN_DOC, _SQL_MANHATTAN_QV,
        (query_doc_id, query_doc_id), (),
        query_doc_id, query_vector, top_n,
        order_desc=False,
    )


# ── SQL views ────────────────────────────────────────────────────────────────

_VIEW_COSINE = """
CREATE OR REPLACE VIEW v_cosine_similarity AS
    SELECT
        a.document_id   AS doc_a_id,
        da.title        AS doc_a_title,
        b.document_id   AS doc_b_id,
        db.title        AS doc_b_title,
        ROUND(
            SUM(a.frequency * b.frequency)
              / (SQRT(SUM(a.frequency * a.frequency))
                 * SQRT(SUM(b.frequency * b.frequency)))
        , 4) AS cosine_score
    FROM HAS a
    JOIN HAS b  ON a.term_id = b.term_id AND b.document_id > a.document_id
    JOIN DOCUMENT da ON da.id = a.document_id
    JOIN DOCUMENT db ON db.id = b.document_id
    GROUP BY a.document_id, b.document_id, da.title, db.title
    ORDER BY cosine_score DESC
"""

_VIEW_EUCLIDEAN = """
CREATE OR REPLACE VIEW v_euclidean_distance AS
    SELECT
        a.document_id   AS doc_a_id,
        da.title        AS doc_a_title,
        b.document_id   AS doc_b_id,
        db.title        AS doc_b_title,
        ROUND(
            SQRT(SUM(POW(a.frequency - b.frequency, 2)))
        , 4) AS euclidean_score
    FROM HAS a
    JOIN HAS b  ON a.term_id = b.term_id AND b.document_id > a.document_id
    JOIN DOCUMENT da ON da.id = a.document_id
    JOIN DOCUMENT db ON db.id = b.document_id
    GROUP BY a.document_id, b.document_id, da.title, db.title
    ORDER BY euclidean_score ASC
"""

_VIEW_NORMS = """
CREATE OR REPLACE VIEW v_document_norms AS
    SELECT
        d.id,
        d.title,
        COUNT(h.term_id)                             AS term_count,
        ROUND(SQRT(SUM(h.frequency * h.frequency)), 4) AS l2_norm,
        ROUND(SUM(h.frequency), 4)                   AS l1_norm
    FROM DOCUMENT d
    JOIN HAS h ON h.document_id = d.id
    GROUP BY d.id, d.title
    ORDER BY l2_norm DESC
"""


def create_views(conn) -> None:
    """Create the three analytical views in the database."""
    cursor = conn.cursor()
    for ddl in (_VIEW_COSINE, _VIEW_EUCLIDEAN, _VIEW_NORMS):
        cursor.execute(ddl)
    conn.commit()
    cursor.close()
