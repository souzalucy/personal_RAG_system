"""Local BM25 vectorless RAG — a drop-in replacement for PageIndex.

Stores document text per file and retrieves relevant passages using
the BM25 ranking algorithm (keyword-based, no embeddings needed).

Uses Lua for tokenization (via bm25_scorer.lua) and checks feature
flags from lua/features.lua for toggling BM25 on/off at runtime.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.lua_runtime import is_enabled, execute as lua_execute

# ---------------------------------------------------------------------------
# Storage: a simple JSON file mapping filename → list of page texts
# ---------------------------------------------------------------------------
_STORAGE_DIR = Path("./data/bm25_storage")
_DOCS_FILE = _STORAGE_DIR / "documents.json"


def _load_docs() -> dict[str, list[str]]:
    """Load stored documents: {filename: [page_text, ...]}"""
    if _DOCS_FILE.exists():
        return json.loads(_DOCS_FILE.read_text())
    return {}


def _save_docs(docs: dict[str, list[str]]) -> None:
    _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _DOCS_FILE.write_text(json.dumps(docs, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Tokenisation helper (delegates to Lua when available)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words.

    Uses Lua's bm25_scorer.lua when available, falls back to Python regex.
    """
    try:
        if is_enabled("bm25_enabled"):
            result = lua_execute("bm25_scorer.lua", "tokenize", text)
            if result and isinstance(result, (list, tuple)):
                return list(result)
    except Exception:
        pass
    # Fallback: simple Python tokenization
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_document(file_path: str, page_texts: list[str]) -> int:
    """Store page texts for a document.

    Args:
        file_path: Path to the PDF file.
        page_texts: List of text strings, one per page.

    Returns:
        Number of pages stored.
    """
    filename = os.path.basename(file_path)
    docs = _load_docs()
    docs[filename] = page_texts
    _save_docs(docs)
    return len(page_texts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 5) -> list[dict]:
    """Search across all stored documents using BM25.

    Checks the 'bm25_enabled' feature flag — if disabled, returns empty.

    Returns a list of dicts with keys: text, page, source, score.
    """
    if not is_enabled("bm25_enabled"):
        return []

    docs = _load_docs()
    if not docs:
        return []

    # Build a flat list of (source, page_no, text) tuples
    corpus: list[tuple[str, int, str]] = []
    for filename, pages in docs.items():
        for i, page_text in enumerate(pages, start=1):
            corpus.append((filename, i, page_text))

    if not corpus:
        return []

    # Tokenise corpus
    tokenized_corpus = [_tokenize(text) for _, _, text in corpus]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Score
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top_k results
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            filename, page_no, text = corpus[idx]
            results.append({
                "text": text,
                "page": page_no,
                "source": filename,
                "score": float(scores[idx]),
            })

    return results


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_documents() -> list[dict]:
    """List documents stored in BM25 index."""
    docs = _load_docs()
    return [
        {"filename": name, "pages": len(pages)}
        for name, pages in docs.items()
    ]


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_document(filename: str) -> bool:
    """Remove a document from the BM25 index."""
    docs = _load_docs()
    if filename in docs:
        del docs[filename]
        _save_docs(docs)
        return True
    return False
