# LSI Document Retrieval System

A MySQL + Python + Streamlit application for Latent Semantic Indexing (LSI) over a PDF document corpus.

## Setup

**Step 1 — Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Configure and initialize MySQL**
```bash
cp .env.example .env
# Edit .env with your MySQL credentials and API keys
mysql -u YOUR_USER -p < schema.sql
```

**Step 3 — Run the application**
```bash
streamlit run app.py
```
