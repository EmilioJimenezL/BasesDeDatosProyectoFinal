# DocBase — LSI Document Retrieval System

A document retrieval system built on **Latent Semantic Indexing (LSI)** for the LIS-3012 Advanced Databases course at UDLAP. The system ingests a corpus of PDF documents (Ley Federal del Trabajo), processes them through a classical NLP pipeline, stores the frequency matrix in MySQL, and exposes a query interface via a local Streamlit web app. An optional LLM layer synthesizes natural-language answers from retrieved documents.

---

## What it does

1. Extracts text from encrypted/indexed PDFs using `pdfplumber`
2. Preprocesses text through a stop-word list, suffix stripping, and word stemming (all stored in MySQL)
3. Builds a term-document frequency matrix (FrecT) and stores it in a relational schema
4. Decomposes FrecT using SVD (`scipy`) — an expert user selects the rank-k cutoff
5. Answers queries by projecting them into LSI space and ranking documents using SQL-computed similarity functions (cosine, Dice, Jaccard, Euclidean)
6. Optionally passes retrieved chunks to a local LLM (Ollama) or Gemini API for a natural-language answer

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.9+ |
| MySQL | 8.x |
| Ollama *(optional)* | latest |

All Python packages are listed in `requirements.txt`.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/your-username/docbase.git
cd docbase
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your MySQL credentials and (optionally) your Gemini API key:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_mysql_user
DB_PASS=your_mysql_password
DB_NAME=docbase
GEMINI_API_KEY=your_gemini_api_key   # optional
OLLAMA_MODEL=llama3.2:3b             # optional
```

### 3. Set up the database

```bash
mysql -u your_user -p < schema.sql
```

This creates the `docbase` database with all 8 tables using `utf8mb4_spanish_ci` collation for correct Spanish character handling.

### 4. Add your PDF documents

Place PDF files in `data/pdfs/`. The system expects at least 10 documents. For the Ley Federal del Trabajo, one PDF per title or chapter works well:

```
data/pdfs/
├── lft_01_titulo_primero.pdf
├── lft_02_titulo_segundo.pdf
├── ...
```

### 5. Run ingestion

```bash
python ingest.py
```

This extracts text from all PDFs, preprocesses it, builds the FrecT matrix, runs SVD, and stores everything in MySQL. Ingestion is idempotent — re-running it skips already-processed documents.

### 6. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Using the app

The Streamlit interface has four tabs:

**Query** — Type a natural-language question. The system preprocesses it, projects it into LSI space, and returns the top-N most similar documents ranked by your chosen similarity function. If Ollama is running (or a Gemini key is set), a synthesized answer appears below the results.

**Matrix viewer** — Inspect the full FrecT frequency matrix and an optional heatmap. Useful for verifying ingestion worked correctly.

**SVD explorer** — View singular values as a bar chart, see explained variance, and change the rank-k cutoff. The system re-queries with the new k without re-running SVD.

**Document browser** — Browse all documents in the corpus, view extracted text, and inspect the term-frequency vector for any document.

### Sidebar controls

| Control | Description |
|---|---|
| Similarity function | Cosine (default), Dice, Jaccard, Euclidean, Manhattan |
| Top-N results | How many documents to return (1–20) |
| SVD rank k | Number of singular values to retain |
| Re-ingest | Trigger a fresh ingestion pass |

---

## Project structure

```
docbase/
├── src/
│   ├── db.py               # MySQL connection pool and query helpers
│   ├── pdf_extractor.py    # PDF text extraction via pdfplumber
│   ├── preprocessor.py     # Stop list, suffix stripping, stemming
│   ├── matrix_builder.py   # FrecT construction and DB insertion
│   ├── svd_engine.py       # SVD decomposition via scipy
│   ├── similarity.py       # SQL similarity and distance functions
│   ├── query_engine.py     # Full query pipeline: preprocess → project → rank
│   └── llm.py              # Ollama / Gemini answer synthesis with fallback
├── data/
│   ├── pdfs/               # Source PDF corpus (not committed)
│   ├── stopwords/          # Spanish stop-word lists
│   └── suffixes/           # Spanish suffix rules
├── tests/                  # Unit tests for similarity functions
├── notebooks/              # Exploratory analysis
├── app.py                  # Streamlit UI entry point
├── ingest.py               # Ingestion pipeline entry point
├── schema.sql              # MySQL DDL for all 8 tables
├── requirements.txt
├── .env.example
└── README.md
```

---

## Database schema

Eight tables store the full pipeline state:

| Table | Purpose |
|---|---|
| `DOCUMENT` | One row per PDF file with metadata |
| `TERM` | Unique stemmed terms after preprocessing |
| `WORD` | Raw word forms that map to each term |
| `HAS` | The FrecT matrix — (document, term, frequency) triples |
| `QUERY` | Saved queries linked to documents |
| `SVD_MATRIX` | Stored T, S, D decomposition values per rank-k |
| `STOP_WORD` | Spanish stop words used during preprocessing |
| `SUFFIX` | Spanish suffix rules for stemming |

The collation `utf8mb4_spanish_ci` is set at the database level so all Spanish characters (á, é, ó, ü, ñ) sort and compare correctly.

---

## Similarity functions

All functions are implemented as SQL queries over the `HAS` table:

| Function | Type | Best for |
|---|---|---|
| Cosine | Similarity | Natural language — angle between vectors |
| Dice | Similarity | Balanced overlap measure |
| Jaccard | Similarity | Penalises poorly shared terms |
| Euclidean | Distance | Raw spatial distance |
| Manhattan | Distance | City-block distance |

---

## LLM layer

The LLM component is optional and the system degrades gracefully without it:

- If **Ollama** is installed and running, the app auto-detects available RAM and selects the best model: `phi3:mini` (4 GB), `llama3.2:3b` (8 GB), or `qwen2.5:7b` (16 GB). `qwen2.5` is preferred for Spanish-language corpora.
- If Ollama is unavailable, the app falls back to the **Gemini 1.5 Flash** API (free tier, requires a key in `.env`).
- If neither is available, the app displays retrieved documents without a synthesized answer — all mandatory retrieval requirements still function.

---

## Running tests

```bash
python -m pytest tests/
```

Tests cover all four SQL similarity functions with known vectors and expected scores.

---

## ABET SO2 compliance

This project was designed to satisfy the ABET Student Outcome 2 (SO2) rubric:

| Criterion | Implementation |
|---|---|
| Identify objectives and constraints | Defined in project report: corpus size, DBMS requirement, similarity functions |
| Analyze the problem | LSI theory, FrecT construction, SVD dimensionality reduction |
| Evaluate solutions | Comparison of similarity metrics; FrecT vs. LSI precision tradeoff |
| Develop solutions | Full pipeline from PDF ingestion to ranked retrieval |
| Implement engineering design | Working system — this repository |
| Non-technical considerations | Labor law access as a social equity and workers' rights issue |

---

## Course information

- Course: LIS-3012 Advanced Databases
- Institution: Universidad de las Américas Puebla (UDLAP)
- Professor: Dr. José Luis Zechinelli Martini

---

## License

MIT
