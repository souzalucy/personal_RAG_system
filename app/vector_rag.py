"""Vector RAG: docling parsing → chunking → Qdrant Cloud storage/retrieval.

Uses Lua scripts (via lupa) for query normalization, hybrid scoring,
and language filtering — all toggleable via lua/features.lua.
"""

import os
from typing import Optional

import numpy as np
from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    QueryResponse,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.lua_runtime import (
    is_enabled,
    get_feature_config,
    execute as lua_execute,
    clear_cache as lua_clear_cache,
)


# ---------------------------------------------------------------------------
# Embedding model (loaded once)
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _model


# ---------------------------------------------------------------------------
# Qdrant client (lazy singleton) — connects to Qdrant Cloud
# ---------------------------------------------------------------------------
_client: Optional[QdrantClient] = None


def _get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        if settings.qdrant_url and settings.qdrant_api_key:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        elif settings.qdrant_url:
            _client = QdrantClient(url=settings.qdrant_url)
        else:
            raise ValueError(
                "Qdrant Cloud requires QDRANT_URL (and QDRANT_API_KEY if the cluster requires authentication). "
                "Set them in your .env file."
            )
    return _client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def _ensure_collection():
    """Create the Qdrant collection if it doesn't exist yet."""
    client = _get_qdrant()
    collections = client.get_collections().collections
    if not any(c.name == settings.qdrant_collection for c in collections):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=768,  # intfloat/multilingual-e5-base dimension
                distance=Distance.COSINE,
            ),
        )


# ---------------------------------------------------------------------------
# Document parsing & chunking
# ---------------------------------------------------------------------------

def parse_document(file_path: str) -> tuple[str, list[dict]]:
    """Parse a PDF with docling and return (full_text, chunks).

    Each chunk dict: {"text": str, "page": int, "source": filename}
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    filename = os.path.basename(file_path)
    chunks: list[dict] = []

    # Use export_to_markdown for full text, then split by pages
    full_text = doc.export_to_markdown()

    # Docling 2.x stores text items with page references
    # We iterate through all text items and group by page
    page_texts: dict[int, list[str]] = {}
    for text_item in doc.texts:
        page_no = text_item.prov[0].page_no if text_item.prov else 1
        if page_no not in page_texts:
            page_texts[page_no] = []
        page_texts[page_no].append(text_item.text)

    for page_no in sorted(page_texts.keys()):
        text = " ".join(page_texts[page_no]).strip()
        if text:
            chunks.append({"text": text, "page": page_no, "source": filename})

    return full_text, chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_document(file_path: str) -> int:
    """Parse a PDF, embed chunks, and store them in Qdrant.

    Returns the number of chunks indexed.
    """
    _ensure_collection()
    embedder = _get_embedder()
    client = _get_qdrant()

    _, chunks = parse_document(file_path)
    if not chunks:
        return 0

    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=abs(hash(f"{chunk['source']}_{chunk['page']}_{i}")),
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                },
            )
        )

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_k: int = 5,
    language: str | None = None,
    source_filter: str | None = None,
) -> list[dict]:
    """Search Qdrant for the most relevant chunks.

    Supports optional Lua-powered features (toggleable via features.lua):
    - language filtering (via filter_language.lua)
    - hybrid scoring (via score_hybrid.lua)
    - query normalization (via normalize_query.lua)

    Args:
        query: The search query.
        top_k: Number of results to return.
        language: Optional language code ("en", "pt") to filter by.
        source_filter: Optional document source to filter by.

    Returns a list of dicts with keys: text, page, source, score.
    """
    _ensure_collection()
    embedder = _get_embedder()
    client = _get_qdrant()

    # --- Query normalization via Lua (toggleable) ---
    normalized_query = query
    if is_enabled("query_normalization"):
        try:
            normalized_query = lua_execute(
                "normalize_query.lua", "normalize", query
            )
        except Exception as exc:
            if is_enabled("debug_logging"):
                print(f"[vector_rag] normalize_query failed: {exc}")
            # fall back to raw query

    query_vector = embedder.encode([f"query: {normalized_query}"])[0].tolist()

    # --- Build filter ---
    qdrant_filter = None
    conditions = []

    if language:
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language))
        )
    if source_filter:
        conditions.append(
            FieldCondition(key="source", match=MatchValue(value=source_filter))
        )

    if conditions:
        qdrant_filter = Filter(must=conditions)

    # --- Search ---
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
    )

    hits = [
        {
            "text": hit.payload.get("text", ""),
            "page": hit.payload.get("page"),
            "source": hit.payload.get("source", ""),
            "score": hit.score,
        }
        for hit in results.points
    ]

    # --- Hybrid scoring via Lua (toggleable) ---
    if is_enabled("hybrid_scoring") and hits:
        try:
            config = get_feature_config("hybrid_scoring")
            keyword_boost = config.get("keyword_boost", 0.15)
            hits = lua_execute(
                "score_hybrid.lua", "rerank", query, hits, keyword_boost
            )
        except Exception as exc:
            if is_enabled("debug_logging"):
                print(f"[vector_rag] hybrid_scoring failed: {exc}")

    return hits


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_indexed_documents() -> list[dict]:
    """List all unique document sources in the Qdrant collection."""
    client = _get_qdrant()
    try:
        collection_info = client.get_collection(settings.qdrant_collection)
    except Exception:
        return []

    # Scroll through all points to collect unique sources
    documents: dict[str, int] = {}
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=next_offset,
            with_payload=["source"],
            with_vectors=False,
        )
        for rec in records:
            src = rec.payload.get("source", "unknown")
            documents[src] = documents.get(src, 0) + 1
        if next_offset is None:
            break

    return [
        {"filename": name, "chunks": count}
        for name, count in documents.items()
    ]


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_document(filename: str) -> bool:
    """Delete all points for a given document source from Qdrant.

    Returns True if any points were deleted, False otherwise.
    """
    client = _get_qdrant()
    try:
        # Use scroll to find all point IDs for this source
        point_ids: list[int] = []
        next_offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            for rec in records:
                if rec.payload.get("source") == filename:
                    point_ids.append(rec.id)
            if next_offset is None:
                break

        if not point_ids:
            return False

        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=point_ids,
        )
        return True
    except Exception:
        return False
