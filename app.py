"""Streamlit web application for the LSI Document Retrieval System."""

import os
import subprocess
import sys
import time

import streamlit as st

# ── Page config MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="DocBase — Ley Federal del Trabajo",
    page_icon="DB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now safe to import the rest
import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

from src import llm
from src.db import get_connection
from src.query_engine import query as run_query, get_all_documents

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pdfs")

METHOD_LABELS = {
    "cosine":    "Coseno (recomendado)",
    "dice":      "Dice",
    "jaccard":   "Jaccard",
    "euclidean": "Euclidiana (distancia)",
    "manhattan": "Manhattan (distancia)",
}
DISTANCE_METHODS = {"euclidean", "manhattan"}


# ── Session state ───────────────────────────────────────────────────────────

def _ensure_conn():
    """Return a healthy MySQL connection, reconnecting if needed."""
    conn = st.session_state.get("conn")
    try:
        if conn is None or not conn.is_connected():
            conn = get_connection()
            st.session_state.conn = conn
    except Exception:
        conn = get_connection()
        st.session_state.conn = conn
    return conn


# ── DB helpers ──────────────────────────────────────────────────────────────

def _safe_fetch(sql: str, params: tuple = (), dict_rows: bool = True):
    """Execute a SELECT and return rows; show st.error and reset conn on failure."""
    try:
        conn = _ensure_conn()
        cursor = conn.cursor(dictionary=dict_rows)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except mysql.connector.Error as e:
        st.error(f"Error de base de datos: {e}")
        st.session_state.conn = None
        if st.button("Reconectar"):
            st.rerun()
        return []


def _resolve_pdf_path(doc_row: dict) -> str:
    """Resolve a DOCUMENT.url to an absolute PDF path on disk."""
    url = doc_row.get("url") or f"{doc_row.get('title', '')}.pdf"
    if os.path.isabs(url) and os.path.exists(url):
        return url
    candidate = os.path.join(PDF_DIR, os.path.basename(url))
    return candidate


def _matched_terms_count(conn, doc_id: int, query_terms: list) -> int:
    """Count how many of *query_terms* (post-preprocessing) exist in HAS for doc_id."""
    if not query_terms:
        return 0
    fmt = ",".join(["%s"] * len(query_terms))
    sql = (
        f"SELECT COUNT(*) AS n FROM HAS h "
        f"JOIN TERM t ON t.id = h.term_id "
        f"WHERE h.document_id = %s AND t.name IN ({fmt})"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, (doc_id, *query_terms))
    row = cursor.fetchone()
    cursor.close()
    return int(row["n"]) if row else 0


def _preprocess_query_terms(text: str, conn) -> list:
    """Re-run the preprocessor to get the stemmed terms used for matching."""
    from src.query_engine import _cache, _load_cache
    from src.preprocessor import preprocess_query
    _load_cache(conn)
    return preprocess_query(text, _cache["stop_words"], _cache["suffix_rules"])


@st.cache_data
def load_pdf_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.title("DocBase")
        st.caption("LIS-3012 · UDLAP")

        st.subheader("Configuración de búsqueda")
        method = st.selectbox(
            "Función de similaridad",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda m: METHOD_LABELS[m],
            key="cfg_method",
        )
        top_n = st.slider("Documentos a mostrar", 1, 10, 5, key="cfg_top_n")
        k_rank = st.slider(
            "Rango SVD (k)", 1, 9, 9,
            help="Número de valores singulares a retener. Máximo=9 para este corpus.",
            key="cfg_k_rank",
        )

        st.divider()
        st.subheader("Estado del sistema")

        doc_count = _safe_fetch("SELECT COUNT(*) AS n FROM DOCUMENT")
        term_count = _safe_fetch("SELECT COUNT(*) AS n FROM TERM")
        has_count = _safe_fetch("SELECT COUNT(*) AS n FROM HAS")

        stats = pd.DataFrame(
            {
                "Métrica": [
                    "Documentos en corpus",
                    "Términos indexados",
                    "Celdas FrecT",
                    "SVD rango k",
                ],
                "Valor": [
                    doc_count[0]["n"] if doc_count else "?",
                    term_count[0]["n"] if term_count else "?",
                    has_count[0]["n"] if has_count else "?",
                    k_rank,
                ],
            }
        )
        st.dataframe(stats, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Modelo de lenguaje")

        available = llm.get_available_models()

        selected_idx = st.selectbox(
            label="Modelo para síntesis",
            options=available['options'],
            index=available['default_index'],
            key='llm_selector'
        )

        sel_provider, sel_model = available['values'][
            available['options'].index(selected_idx)
        ]

        st.session_state['llm_provider'] = sel_provider
        st.session_state['llm_model'] = sel_model

        st.divider()
        st.subheader("Herramientas")
        if st.button("Re-indexar corpus", use_container_width=True):
            with st.spinner("Re-indexando..."):
                result = subprocess.run(
                    [sys.executable, "ingest.py"],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                st.success("Re-indexación completada.")
                st.session_state.conn = None
            else:
                st.error(f"Error: {result.stderr[:300]}")

    return {"method": method, "top_n": top_n, "k_rank": k_rank}


# ── Tab 1 — Consulta ────────────────────────────────────────────────────────

def render_query_tab(cfg: dict):
    st.header("Consulta en lenguaje natural")

    with st.form(key='search_form'):
        query_input = st.text_input(
            label="Consulta",
            placeholder="Ejemplo: jornada laboral salario mínimo",
            key="query_input"
        )
        search_submitted = st.form_submit_button(
            label="Buscar",
            type="primary",
            use_container_width=False
        )

    if not (search_submitted and query_input.strip()):
        return

    conn = _ensure_conn()

    progress = st.progress(0, text="Buscando documentos...")
    progress.progress(25, text="Preprocesando consulta...")

    try:
        results = run_query(
            query_input, conn,
            method=cfg["method"], top_n=cfg["top_n"], k=cfg["k_rank"],
        )
        progress.progress(75, text="Calculando similitud...")
        progress.progress(100, text="Búsqueda completada.")
    except ValueError as e:
        progress.empty()
        st.error(str(e))
        st.stop()
    except mysql.connector.Error as e:
        progress.empty()
        st.error(f"Error de base de datos: {e}")
        st.session_state.conn = None
        st.stop()

    time.sleep(0.3)
    progress.empty()

    if not results:
        st.warning("No se encontraron documentos.")
        return

    query_terms = _preprocess_query_terms(query_input, conn)
    total_terms = len(query_terms)

    provider = st.session_state.get('llm_provider', 'none')
    model = st.session_state.get('llm_model', None)

    if provider != 'none':
        st.subheader("Síntesis")
        llm_progress = st.progress(0, text="Iniciando modelo...")
        llm_progress.progress(30, text="Analizando fragmentos relevantes...")

        results_for_llm = []
        for r in results:
            doc_rows = _safe_fetch(
                "SELECT id, url, title FROM DOCUMENT WHERE id = %s",
                (r["document_id"],),
            )
            url = doc_rows[0]["url"] if doc_rows else None
            results_for_llm.append({
                "document_id": r["document_id"],
                "title": r["title"],
                "url": _resolve_pdf_path({"url": url, "title": r["title"]}),
            })

        answer = llm.synthesize(
            query_input, results_for_llm,
            provider=provider,
            model=model,
            query_tokens=query_terms
        )

        llm_progress.progress(100, text="Síntesis completada.")
        time.sleep(0.3)
        llm_progress.empty()

        if answer:
            st.markdown(answer)
            st.caption(f"Generado por {model} · basado en los documentos recuperados")

        st.divider()

    is_distance = cfg["method"] in DISTANCE_METHODS
    for i, r in enumerate(results):
        doc_rows = _safe_fetch(
            "SELECT id, url, title FROM DOCUMENT WHERE id = %s",
            (r["document_id"],),
        )
        pdf_path = _resolve_pdf_path(doc_rows[0]) if doc_rows else None

        with st.container(border=True):
            col_score, col_title = st.columns([1, 5])
            with col_score:
                st.metric(
                    label=f"#{i+1}",
                    value=f"{r['score']:.4f}"
                )
            with col_title:
                st.markdown(f"**{r['title']}**")
                matched = _matched_terms_count(conn, r["document_id"], query_terms)
                st.caption(f"Términos coincidentes: {matched} / {total_terms}")

                if pdf_path:
                    passage = llm.get_document_text(
                        pdf_path, query_tokens=query_terms, max_chars=400
                    )
                    if passage:
                        st.caption("Fragmento relevante:")
                        st.markdown(
                            f"> {passage}",
                            help="Fragmento con mayor coincidencia con la consulta"
                        )

            if pdf_path and os.path.exists(pdf_path):
                pdf_bytes = load_pdf_bytes(pdf_path)
                st.download_button(
                    label=f"Descargar {r['title']}",
                    data=pdf_bytes,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key=f"download_{r['document_id']}_{i}"
                )

            st.divider()


# ── Tab 2 — Matriz FrecT ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_frect_matrix() -> pd.DataFrame:
    conn = _ensure_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT t.name AS term, d.title AS document, h.frequency
        FROM HAS h
        JOIN TERM t ON t.id = h.term_id
        JOIN DOCUMENT d ON d.id = h.document_id
    """)
    rows = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Truncate document titles to 25 chars for compact column headers
    df["document"] = df["document"].str.slice(0, 25)
    pivot = df.pivot(index="term", columns="document", values="frequency").fillna(0.0)
    return pivot


def render_frect_tab():
    st.header("Matriz de frecuencias TF-IDF (FrecT)")
    st.caption("Cada celda muestra el peso TF-IDF del término en el documento.")

    df = _load_frect_matrix()
    if df.empty:
        st.warning("No hay datos en HAS.")
        return

    st.caption(f"{len(df)} términos × {len(df.columns)} documentos")
    st.dataframe(df, use_container_width=True)

    st.divider()
    top_terms_n = st.slider(
        "Mostrar top N términos por frecuencia total",
        min_value=10, max_value=100, value=30,
    )
    row_sums = df.sum(axis=1).sort_values(ascending=False)
    filtered = df.loc[row_sums.head(top_terms_n).index]
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Mapa de calor")
    fig, ax = plt.subplots(figsize=(12, max(6, top_terms_n * 0.25)))
    sns.heatmap(filtered, ax=ax, cmap="YlOrRd", fmt=".3f")
    ax.set_title("FrecT — Pesos TF-IDF")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)


# ── Tab 3 — Explorador SVD ──────────────────────────────────────────────────

def render_svd_tab():
    st.header("Descomposición SVD")

    s_rows = _safe_fetch("SELECT DISTINCT s_value, k_rank FROM SVD_MATRIX")
    s_values = sorted([float(r["s_value"]) for r in s_rows], reverse=True)
    if not s_values:
        st.warning("No hay valores singulares en SVD_MATRIX.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Valores singulares")
        fig, ax = plt.subplots()
        ax.bar(range(1, len(s_values) + 1), s_values, color="steelblue")
        ax.set_xlabel("Componente k")
        ax.set_ylabel("Valor singular")
        ax.set_title("Distribución de valores singulares")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Varianza explicada acumulada")
        total = sum(s ** 2 for s in s_values)
        cumvar = [
            sum(s ** 2 for s in s_values[: i + 1]) / total * 100
            for i in range(len(s_values))
        ]
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="coral")
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.7, label="80% threshold")
        ax.set_xlabel("Número de componentes k")
        ax.set_ylabel("Varianza explicada (%)")
        ax.set_title("Varianza explicada acumulada")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Matriz de similaridad entre documentos (coseno)")
    pair_rows = _safe_fetch(
        "SELECT doc_a_title, doc_b_title, cosine_score FROM v_cosine_similarity"
    )
    if pair_rows:
        all_titles = sorted({r["doc_a_title"] for r in pair_rows}
                            | {r["doc_b_title"] for r in pair_rows})
        sim_df = pd.DataFrame(1.0, index=all_titles, columns=all_titles)
        best = (None, None, -1.0)
        for r in pair_rows:
            a, b, s = r["doc_a_title"], r["doc_b_title"], float(r["cosine_score"])
            sim_df.at[a, b] = s
            sim_df.at[b, a] = s
            if s > best[2]:
                best = (a, b, s)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim_df, ax=ax, cmap="viridis", annot=True, fmt=".2f",
                    cbar_kws={"label": "Cosine similarity"})
        ax.set_title("Document-document cosine similarity")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Par más similar: {best[0]} ↔ {best[1]} ({best[2]:.4f})")

    st.info("""
    **¿Qué significa k?**
    k controla cuántos "conceptos latentes" retiene el modelo.
    - k pequeño: captura solo los temas más dominantes
    - k grande: captura más variación pero puede incluir ruido
    - Para este corpus de 10 documentos, k=9 es el máximo matemático
    """)


# ── Tab 4 — Documentos ──────────────────────────────────────────────────────

def render_documents_tab():
    st.header("Corpus de documentos")
    conn = _ensure_conn()

    docs = get_all_documents(conn)
    st.metric("Total documentos", len(docs))

    for doc in docs:
        with st.expander(doc["title"]):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Archivo:** {doc.get('url', '?')}")
                st.write(f"**Autor:** {doc.get('author', '?')}")
                st.write(f"**Fecha:** {doc.get('doc_date', '?')}")
                st.write("**Fragmento:**")
                pdf_path = _resolve_pdf_path(doc)
                snippet = llm.get_document_text(pdf_path, max_chars=600)
                st.text(snippet or "(no se pudo extraer texto)")

            with col2:
                stats = _safe_fetch("""
                    SELECT COUNT(*) AS term_count,
                           SUM(frequency) AS total_weight,
                           MAX(frequency) AS max_weight
                    FROM HAS WHERE document_id = %s
                """, (doc["id"],))
                if stats:
                    s = stats[0]
                    st.metric("Términos únicos", int(s["term_count"] or 0))
                    st.metric("Peso total", f"{float(s['total_weight'] or 0):.4f}")
                    st.metric("Peso máximo", f"{float(s['max_weight'] or 0):.4f}")

                st.write("**Top 5 términos:**")
                top_terms = _safe_fetch("""
                    SELECT t.name, h.frequency
                    FROM HAS h JOIN TERM t ON t.id = h.term_id
                    WHERE h.document_id = %s
                    ORDER BY h.frequency DESC LIMIT 5
                """, (doc["id"],))
                if top_terms:
                    st.dataframe(
                        pd.DataFrame(top_terms),
                        use_container_width=True,
                        hide_index=True,
                    )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    cfg = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Consulta",
        "Matriz FrecT",
        "Explorador SVD",
        "Documentos",
    ])
    with tab1:
        render_query_tab(cfg)
    with tab2:
        render_frect_tab()
    with tab3:
        render_svd_tab()
    with tab4:
        render_documents_tab()


main()
