"""Document ingestion script.

CLI entry point that reads PDFs from data/pdfs/, extracts and preprocesses text,
builds the term-document matrix, runs SVD, and persists everything to MySQL.
"""

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src.db import get_db, execute_query
from src.pdf_extractor import extract_all
from src.preprocessor import load_stop_words, load_suffix_rules, preprocess
from src.matrix_builder import (
    build_frequency_matrix,
    apply_tfidf,
    save_terms,
    save_matrix,
)
from src.svd_engine import compute_svd, save_svd


def _step(label: str):
    """Print a step label and return the start time."""
    print(label)
    return time.time()


def _done(t0: float):
    """Print elapsed time since t0."""
    print(f"  → completed in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="LSI document ingestion pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate TERM, WORD, HAS, SVD_MATRIX tables before ingesting",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="SVD rank to compute (default: 100)",
    )
    args = parser.parse_args()

    try:
        # ── Step 1 ─────────────────────────────────────────────────────
        t0 = _step("[1/6] Connecting to MySQL...")
        with get_db() as conn:
            _done(t0)

            if args.reset:
                print("  ⚠  --reset: truncating TERM, WORD, HAS, SVD_MATRIX...")
                cursor = conn.cursor()
                # Truncate in FK-safe order
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                for table in ("SVD_MATRIX", "HAS", "WORD", "TERM"):
                    cursor.execute(f"TRUNCATE TABLE {table}")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                conn.commit()
                cursor.close()
                print("  → tables truncated")

            # ── Step 2 ─────────────────────────────────────────────────
            t0 = _step("[2/6] Loading preprocessing rules...")
            stop_words = load_stop_words(conn)
            suffix_rules = load_suffix_rules(conn)
            print(f"  → {len(stop_words)} stop words, {len(suffix_rules)} suffix rules")
            _done(t0)

            # ── Step 3 ─────────────────────────────────────────────────
            t0 = _step("[3/6] Extracting text from PDFs...")
            raw_texts = extract_all("data/pdfs")
            print(f"  → {len(raw_texts)} documents extracted")
            _done(t0)

            # Preprocess each document
            print("  → Preprocessing documents...")
            t_pre = time.time()
            processed: dict[str, list[str]] = {}
            for name, text in raw_texts.items():
                tokens = preprocess(text, stop_words, suffix_rules)
                processed[name] = tokens
            print(f"  → preprocessing completed in {time.time() - t_pre:.1f}s")

            # ── Step 4 ─────────────────────────────────────────────────
            t0 = _step("[4/6] Building frequency matrix...")
            term_index, doc_index, matrix = build_frequency_matrix(processed)
            print(f"  → {len(term_index)} terms × {len(doc_index)} documents")
            matrix = apply_tfidf(matrix, term_index, doc_index)
            print("  → TF-IDF weights applied")
            _done(t0)

            # ── Step 5 ─────────────────────────────────────────────────
            k = args.k
            t0 = _step(f"[5/6] Computing SVD (k={k})...")
            U, s, Vt = compute_svd(matrix, k=k)
            actual_k = len(s)
            if actual_k < k:
                print(f"  → k clamped to {actual_k} (matrix too small for k={k})")
                k = actual_k
            _done(t0)

            # ── Step 6 ─────────────────────────────────────────────────
            t0 = _step("[6/6] Saving to database...")

            # Build mapping from doc name → DB id
            doc_rows = execute_query(
                conn, "SELECT id, url FROM DOCUMENT"
            )
            # url stores the filename; stem is the name without extension
            doc_db_ids: dict[str, int] = {}
            for row in doc_rows:
                stem = row["url"].rsplit(".", 1)[0] if "." in row["url"] else row["url"]
                doc_db_ids[stem] = row["id"]

            # Save terms and get DB ids
            term_db_ids = save_terms(conn, term_index)
            print(f"  → {len(term_db_ids)} terms saved")

            # Save frequency matrix
            cells = save_matrix(
                conn, matrix, term_index, doc_index, term_db_ids, doc_db_ids
            )
            print(f"  → {cells} matrix cells saved")

            # Save SVD
            save_svd(
                conn, U, s, Vt, term_index, doc_index,
                term_db_ids, doc_db_ids, k
            )
            print(f"  → SVD saved at k={k}")
            _done(t0)

            print(
                f"\nDone. {len(term_db_ids)} terms, {len(doc_index)} documents, "
                f"{cells} matrix cells, SVD stored at k={k}."
            )

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
