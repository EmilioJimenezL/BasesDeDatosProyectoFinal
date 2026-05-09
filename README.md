# LSI Document Retrieval System

A document search engine built on **Latent Semantic Indexing (LSI)**. PDFs are ingested into MySQL, decomposed via truncated SVD, and queried through a Streamlit web UI. An optional LLM layer (Google Gemini or local Ollama) generates natural-language answers from the top results.

---

## How it works

```
PDFs → extract text → preprocess → term-document matrix → SVD → MySQL
                                                                    ↓
                                              query → LSI space → cosine similarity → ranked results → LLM answer
```

1. **Ingestion** (`ingest.py`): reads PDFs, extracts text with `pdfplumber`, tokenizes, removes stop words, applies suffix rules (stored in MySQL), then writes term frequencies to the `HAS` table.
2. **SVD** (`src/svd_engine.py`): builds a sparse term-document matrix and runs `scipy.sparse.linalg.svds` at rank *k*. The T, S, D factor matrices are stored in `SVD_MATRIX`.
3. **Query** (`src/query_engine.py`): preprocesses the query string the same way as documents, projects it into the *k*-dimensional concept space, and ranks documents by cosine similarity.
4. **UI** (`app.py`): Streamlit interface — enter a query, see ranked results with scores, optionally get an LLM-generated answer.

---

## Project structure

```
.
├── app.py                  # Streamlit web application
├── ingest.py               # CLI ingestion script
├── schema.sql              # Full MySQL schema (run once to initialize)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── src/
│   ├── db.py               # MySQL connection factory + CRUD helpers
│   ├── pdf_extractor.py    # PDF text + metadata extraction (pdfplumber)
│   ├── preprocessor.py     # Tokenization, stop-word removal, stemming
│   ├── matrix_builder.py   # Sparse term-document matrix (scipy.sparse)
│   ├── svd_engine.py       # Truncated SVD via scipy + persistence to DB
│   ├── similarity.py       # Cosine similarity in LSI concept space
│   ├── query_engine.py     # End-to-end query pipeline
│   └── llm.py              # Gemini / Ollama answer generation
│
├── data/
│   ├── pdfs/               # Drop input PDFs here (git-ignored)
│   ├── stopwords/          # Optional flat stop-word files for bulk import
│   └── suffixes/           # Optional suffix/replacement files for bulk import
│
├── tests/
│   └── __init__.py
└── notebooks/              # Jupyter exploration notebooks
```

---

## Database schema

MySQL 8+, database `docbase` (`utf8mb4 / utf8mb4_spanish_ci`).

| Table | Purpose |
|-------|---------|
| `DOCUMENT` | One row per ingested PDF — url, title, author, date |
| `TERM` | Unique normalized terms (stems) across the corpus |
| `WORD` | Raw word forms that map to each term |
| `HAS` | Term frequency per document — the raw term-document matrix |
| `SVD_MATRIX` | Stored T/S/D factor values from the SVD decomposition |
| `QUERY` | Persisted query history, optionally linked to a top document |
| `STOP_WORD` | Words excluded from indexing (loaded at preprocessing time) |
| `SUFFIX` | Suffix → replacement rules used for stemming |

**Key relationships:**
- `WORD.term_id → TERM.id` (cascade delete)
- `HAS.(document_id, term_id)` composite PK with cascade deletes on both sides
- `SVD_MATRIX.(term_id, document_id)` both cascade on delete
- `QUERY.document_id → DOCUMENT.id` sets NULL on delete

---

## Prerequisites

- Python 3.9+
- MySQL 8.0+ (running locally or remotely)
- *(Optional)* [Ollama](https://ollama.com) with `llama3.2:3b` pulled, for local LLM answers
- *(Optional)* A Google Gemini API key, for cloud LLM answers

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ProyectoFinal
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_mysql_user
DB_PASS=your_mysql_password
DB_NAME=docbase
GEMINI_API_KEY=your_gemini_api_key   # leave blank to skip Gemini
OLLAMA_MODEL=llama3.2:3b             # change or leave blank to skip Ollama
```

### 4. Initialize the database

```bash
mysql -u your_mysql_user -p < schema.sql
```

This creates the `docbase` database and all tables. Safe to re-run — all statements use `IF NOT EXISTS`.

### 5. Ingest documents

Drop PDF files into `data/pdfs/`, then run:

```bash
python ingest.py
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Environment variables reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DB_HOST` | Yes | MySQL host (default `localhost`) |
| `DB_PORT` | Yes | MySQL port (default `3306`) |
| `DB_USER` | Yes | MySQL username |
| `DB_PASS` | Yes | MySQL password |
| `DB_NAME` | Yes | Database name (default `docbase`) |
| `GEMINI_API_KEY` | No | Google Gemini API key for cloud LLM answers |
| `OLLAMA_MODEL` | No | Ollama model tag (e.g. `llama3.2:3b`) for local LLM answers |

---

## Development notes

- The `.venv/` directory is git-ignored. Every contributor runs `pip install -r requirements.txt` in their own venv.
- `data/pdfs/` is git-ignored — do not commit PDFs to the repo.
- `.env` is git-ignored — never commit credentials.
- The `SUFFIX` table drives stemming: each row maps a suffix string to its replacement (empty string = strip). Load your own rules via `ingest.py` or direct SQL.
- The SVD rank *k* controls the trade-off between precision and recall. A good starting point is 100–300 for a medium corpus; tune by inspecting retrieval quality.
- Both LLM backends are optional. If neither `GEMINI_API_KEY` nor `OLLAMA_MODEL` is set, the app operates in retrieval-only mode.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `mysql-connector-python` | MySQL driver |
| `pdfplumber` | PDF text extraction |
| `scipy` | Sparse SVD (`svds`) |
| `numpy` | Numerical operations |
| `pandas` | Tabular data handling |
| `python-dotenv` | `.env` file loading |
| `requests` | HTTP utilities |
| `psutil` | System resource monitoring |
| `google-generativeai` | Google Gemini LLM |
| `ollama` | Local Ollama LLM client |
