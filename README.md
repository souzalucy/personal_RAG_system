# MAID — My Assistant Integrated Device

**MAID** is a dual Retrieval-Augmented Generation (RAG) system that combines **semantic vector search** (via Qdrant Cloud) with **keyword-based BM25 search** (vectorless, local) to answer questions about your PDF documents. It uses a **Groq LLM** to generate answers from the retrieved context.

The system is fully bilingual (English/Portuguese) and features a **Lua scripting engine** that lets you modify search behavior, scoring, prompts, and feature flags at runtime — no redeployment needed.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Streamlit UI      │  port 8501
                    │  (pastel pink theme)│
                    └──────────┬──────────┘
                               │ HTTP (httpx)
                    ┌──────────▼──────────┐
                    │   FastAPI Backend   │  port 8000
                    │    (app/main.py)    │
                    └──┬───────┬───────┬──┘
                       │       │       │
              ┌────────▼──┐ ┌──▼────┐ ┌▼──────────┐
              │ Vector RAG│ │ BM25  │ │ Lua Engine│
              │ (Qdrant)  │ │(local)│ │ (lupa)    │
              └───────────┘ └───────┘ └───────────┘
```

### Two RAG pipelines

| Pipeline | Method | Storage | Strengths |
|----------|--------|---------|-----------|
| **Vector RAG** | Semantic embeddings (`intfloat/multilingual-e5-base`) | Qdrant Cloud | Understands meaning, synonyms, concepts |
| **BM25** | Keyword frequency (TF-IDF variant) | Local JSON (`data/bm25_storage/`) | Exact keyword matches, bilingual support |

Results from both pipelines are **fused** using Reciprocal Rank Fusion (RRF) — configurable via Lua.

---

## Quick Start

### Prerequisites

- Python 3.14+
- A [Groq API key](https://console.groq.com) (free tier works)
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier available)

### 1. Clone and setup

```bash
git clone <repo-url> && cd RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp deploy/.env.example .env
```

Edit `.env`:

```ini
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=documents
```

### 3. Run

**Option A — Local (two terminals):**

```bash
# Terminal 1: API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

**Option B — Docker Compose:**

```bash
docker compose -f deploy/docker-compose.yml up --build
```

**Option C — Systemd (Linux):**

```bash
sudo cp deploy/rag-api.service /etc/systemd/system/
sudo cp deploy/rag-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-api rag-frontend
```

### 4. Ingest documents

**Via the UI:** Open `http://localhost:8501`, upload a PDF in the sidebar, click "Ingest Document".

**Via CLI:**

```bash
python scripts/ingest.py path/to/document.pdf
python scripts/ingest.py path/to/folder/with/pdfs/
```

### 5. Ask questions

Type your question in the chat input at `http://localhost:8501`. The system searches both RAG pipelines, fuses the results, and generates an answer via Groq.

---

## Project Structure

```
RAG/
├── app/                        # FastAPI backend
│   ├── main.py                 # API routes (query, ingest, list, delete)
│   ├── config.py               # Settings from .env (Pydantic)
│   ├── models.py               # Pydantic request/response models
│   ├── vector_rag.py           # Docling → Qdrant pipeline
│   ├── bm25_rag.py             # Local BM25 keyword search
│   ├── llm.py                  # Groq LLM integration
│   └── lua_runtime.py          # Lua engine (lupa) — script loader & feature flags
├── frontend/
│   ├── streamlit_app.py        # Streamlit UI (pastel pink theme)
│   └── src/
│       ├── catmaid.png         # Header logo (cat)
│       └── personagem_maid.png # Sidebar character image
├── lua/                        # Lua scripts — editable at runtime
│   ├── features.lua            # Central feature flags (hot-reloadable)
│   ├── fusion_strategy.lua     # RRF & concatenate fusion strategies
│   ├── normalize_query.lua     # Query normalization (lowercase, accents, punctuation)
│   ├── score_hybrid.lua        # Hybrid scoring: vector + keyword boost
│   ├── prompt_builder.lua      # LLM prompt construction with token budget
│   └── bm25_scorer.lua         # Tokenization helper for BM25
├── scripts/
│   ├── ingest.py               # CLI tool for batch PDF ingestion
│   └── run_frontend.py         # Streamlit launcher (Python 3.14 compat fix)
├── deploy/
│   ├── docker-compose.yml      # Docker Compose (API + Frontend)
│   ├── Dockerfile.api          # API container
│   ├── Dockerfile.frontend     # Frontend container
│   ├── .env.example            # Environment template
│   ├── rag-api.service         # Systemd service for API
│   ├── rag-frontend.service    # Systemd service for Frontend
│   └── start-rag.sh            # Startup script for systemd
├── data/
│   ├── bm25_storage/           # BM25 document index (JSON)
│   └── qdrant_storage/         # Local Qdrant data (if using local mode)
├── requirements.txt
└── .env                        # Your configuration (git-ignored)
```

---

## Editable Parts

### Frontend (`frontend/streamlit_app.py`)

The UI is built with **Streamlit** and uses a pastel pink theme defined via CSS in the `<style>` block at the top of the file.

**What you can change:**

| What | Where | How |
|------|-------|-----|
| Colors | CSS `<style>` block (lines ~33–106) | Edit hex values for backgrounds, text, buttons, etc. |
| Header logo | `catmaid_path = SRC_DIR / "catmaid.png"` | Replace the image file or change the path |
| Sidebar character | `personagem_path = SRC_DIR / "personagem_maid.png"` | Replace the image file or change the path |
| Page title/icon | `st.set_page_config(...)` | Change `page_title` and `page_icon` |
| Branding text | Header columns (lines ~114–124) | Edit "MAID" and "My Assistant Integrated Device" |
| Layout | Streamlit components throughout | Add/remove columns, expanders, sections |

### Lua Scripts (`lua/`)

Lua scripts are loaded at runtime via **lupa** (LuaJIT bindings). Scripts loaded as **feature flags** (`features.lua`) are re-read from disk on every request — changes take effect immediately. Other scripts are cached but can be reloaded by calling the API's clear cache endpoint or restarting the server.

| Script | Purpose | Editable behavior |
|--------|---------|-------------------|
| `features.lua` | Central feature flags | Toggle BM25, fusion, hybrid scoring, query normalization, document deletion, debug logging — all at runtime |
| `fusion_strategy.lua` | How vector + BM25 results are combined | Change RRF constant `k`, add new fusion strategies |
| `normalize_query.lua` | Query preprocessing | Add/remove normalization steps (stemming, stop words, etc.) |
| `score_hybrid.lua` | Re-rank vector results with keyword boost | Adjust `keyword_boost` weight, change scoring formula |
| `prompt_builder.lua` | LLM system prompt & context formatting | Rewrite the system prompt, change context truncation logic, modify token budget |
| `bm25_scorer.lua` | Tokenization for BM25 | Change tokenization rules |

**Example — disable BM25 at runtime:**

Edit `lua/features.lua` and set:

```lua
bm25_enabled = {
    enabled = false,
},
```

The next API request will skip BM25 search — no restart needed.

### Backend (`app/`)

| File | What it does | Editable |
|------|-------------|----------|
| `config.py` | Environment variables | Add new settings, change defaults |
| `main.py` | API routes | Add/modify endpoints, change query pipeline |
| `vector_rag.py` | Docling parsing, Qdrant operations | Change chunking strategy, embedding model, search parameters |
| `bm25_rag.py` | Local BM25 index | Change storage format, scoring parameters |
| `llm.py` | Groq integration | Change model, temperature, max tokens |
| `models.py` | Pydantic models | Add/change request/response fields |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/query` | Ask a question (body: `{"question": "...", "top_k": 3}`) |
| `POST` | `/api/ingest` | Ingest a PDF (`?file_path=/path/to/doc.pdf`) |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{filename}` | Delete a document (requires `document_deletion.enabled = true` in `features.lua`) |

---

## Feature Flags (`lua/features.lua`)

All flags are checked on every request — edit the file and changes apply immediately.

| Flag | Default | Description |
|------|---------|-------------|
| `hybrid_scoring.enabled` | `true` | Boost vector results with keyword match ratio |
| `hybrid_scoring.keyword_boost` | `0.15` | Weight of keyword matching in hybrid score |
| `language_filter.enabled` | `true` | Enable language detection & filtering |
| `query_normalization.enabled` | `true` | Normalize queries (lowercase, remove accents) |
| `bm25_enabled.enabled` | `true` | Enable/disable BM25 keyword search |
| `fusion.enabled` | `true` | Enable result fusion (RRF or concatenate) |
| `fusion.strategy` | `"reciprocal_rank"` | Fusion strategy: `"reciprocal_rank"` or `"concatenate"` |
| `fusion.reciprocal_rank_k` | `60` | RRF constant (higher = more weight to lower ranks) |
| `document_deletion.enabled` | `false` | Allow document deletion via API |
| `debug_logging.enabled` | `false` | Print debug info to stdout |

---

## Deployment

### Docker

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

### Systemd (Linux)

```bash
# Edit paths in deploy/*.service files if needed, then:
sudo cp deploy/rag-api.service /etc/systemd/system/
sudo cp deploy/rag-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-api rag-frontend
```

### Manual

```bash
# API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (with Python 3.14 compat fix)
python scripts/run_frontend.py
```

---

## Tech Stack

- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit, httpx
- **Vector RAG:** Docling (PDF parsing), Qdrant Cloud (vector DB), Sentence Transformers (embeddings)
- **BM25:** rank-bm25 (local keyword search)
- **LLM:** Groq API (llama-3.3-70b-versatile)
- **Lua Runtime:** lupa (LuaJIT bindings)
- **Infrastructure:** Docker, Docker Compose, systemd
