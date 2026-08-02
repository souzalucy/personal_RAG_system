"""FastAPI application for the dual RAG system.

Uses Lua scripts (via lupa) for:
- Feature flag checks (lua/features.lua) — hot-reloadable
- Source fusion strategy (lua/fusion_strategy.lua)
- Query normalization (lua/normalize_query.lua)
- Hybrid scoring (lua/score_hybrid.lua)
- Prompt building (lua/prompt_builder.lua)
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import QueryRequest, QueryResponse, Source, IngestResponse, FolderIngestResponse, DocumentInfo
from app import vector_rag, bm25_rag, llm
from app.lua_runtime import (
    is_enabled,
    get_feature_config,
    execute as lua_execute,
)

app = FastAPI(title="Dual RAG System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query both RAG systems and return an LLM-generated answer.

    Pipeline (each step checks feature flags from lua/features.lua):
      1. Vector RAG search (semantic)
      2. BM25 search (keyword-based, toggleable)
      3. Fusion strategy (RRF or concatenate, toggleable)
      4. LLM answer generation
    """
    # 1. Vector RAG search (semantic)
    vector_results = vector_rag.search(req.question, top_k=req.top_k)

    # 2. BM25 search (keyword-based, toggleable via features.lua)
    bm25_results = bm25_rag.search(req.question, top_k=req.top_k)

    # 3. Fusion: combine results via Lua strategy (toggleable)
    if is_enabled("fusion") and vector_results and bm25_results:
        config = get_feature_config("fusion")
        strategy = config.get("strategy", "reciprocal_rank")
        k = config.get("reciprocal_rank_k", 60)
        try:
            fused = lua_execute(
                "fusion_strategy.lua",
                "fuse",
                strategy,
                vector_results,
                bm25_results,
                k,
            )
            # Convert fused results to Source objects
            sources = [
                Source(
                    method=item.get("method", "hybrid"),
                    content=item.get("content", ""),
                    document=item.get("source", ""),
                    page=item.get("page"),
                    score=item.get("score"),
                )
                for item in fused
            ]
        except Exception as exc:
            if is_enabled("debug_logging"):
                print(f"[main] fusion failed: {exc}")
            # Fallback: concatenate without Lua
            sources = _concatenate_sources(vector_results, bm25_results)
    else:
        # No fusion: just concatenate
        sources = _concatenate_sources(vector_results, bm25_results)

    if not sources:
        return QueryResponse(
            answer="No documents indexed yet. Please ingest a PDF first.",
            sources=[],
        )

    # 4. Ask LLM with all contexts
    contexts_for_llm = [
        {"method": s.method, "content": s.content, "source": s.document, "page": s.page}
        for s in sources
    ]
    answer = llm.ask(req.question, contexts_for_llm)

    return QueryResponse(answer=answer, sources=sources)


def _concatenate_sources(
    vector_results: list[dict], bm25_results: list[dict]
) -> list[Source]:
    """Simple concatenation fallback when Lua fusion is disabled or fails."""
    sources: list[Source] = []
    seen = set()

    for r in vector_results:
        key = (r.get("source", ""), r.get("text", "")[:200])
        seen.add(key)
        sources.append(
            Source(
                method="vector_rag",
                content=r["text"],
                document=r["source"],
                page=r["page"],
                score=r["score"],
            )
        )

    for r in bm25_results:
        key = (r.get("source", ""), r.get("text", "")[:200])
        if key not in seen:
            seen.add(key)
            sources.append(
                Source(
                    method="bm25",
                    content=r["text"],
                    document=r["source"],
                    page=r["page"],
                    score=r["score"],
                )
            )

    return sources


def _ingest_single(file_path: str) -> IngestResponse:
    """Ingest a single PDF into both RAG systems. Returns response even on partial failure."""
    path = Path(file_path)
    filename = path.name
    chunk_count = 0

    # Vector RAG
    try:
        chunk_count = vector_rag.index_document(str(path))
        vector_status = f"indexed ({chunk_count} chunks)"
    except Exception as e:
        vector_status = f"failed: {e}"

    # BM25 (vectorless) — reuses the same docling parse to get page texts
    try:
        _, page_texts = vector_rag.parse_document(str(path))
        page_count = bm25_rag.index_document(str(path), [p["text"] for p in page_texts])
        bm25_status = f"indexed ({page_count} pages)"
    except Exception as e:
        bm25_status = f"failed: {e}"

    return IngestResponse(
        status="success",
        document=filename,
        vector_rag=vector_status,
        bm25=bm25_status,
        chunks=chunk_count,
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(file_path: str):
    """Ingest a single PDF file into both RAG systems."""
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    return _ingest_single(file_path)


@app.post("/api/ingest-folder", response_model=FolderIngestResponse)
def ingest_folder(folder_path: str):
    """Ingest all PDF files from a folder into both RAG systems."""
    path = Path(folder_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder_path}")

    pdfs = sorted(path.glob("*.pdf")) + sorted(path.glob("*.PDF"))
    if not pdfs:
        raise HTTPException(status_code=404, detail=f"No PDF files found in {folder_path}")

    results: list[IngestResponse] = []
    succeeded = 0
    failed = 0

    for pdf_path in pdfs:
        try:
            result = _ingest_single(str(pdf_path))
            results.append(result)
            succeeded += 1
        except Exception as e:
            results.append(IngestResponse(
                status="failed",
                document=pdf_path.name,
                vector_rag=f"error: {e}",
                bm25=f"error: {e}",
                chunks=0,
            ))
            failed += 1

    return FolderIngestResponse(
        status="success",
        folder=folder_path,
        total_pdfs=len(pdfs),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    """List all indexed documents from both RAG systems."""
    vector_docs = vector_rag.list_indexed_documents()
    bm25_docs = bm25_rag.list_documents()

    # Merge by filename
    merged: dict[str, DocumentInfo] = {}
    for d in vector_docs:
        merged[d["filename"]] = DocumentInfo(
            filename=d["filename"],
            vector_rag_indexed=True,
            bm25_indexed=False,
            chunks=d["chunks"],
        )
    for d in bm25_docs:
        fn = d["filename"]
        if fn in merged:
            merged[fn].bm25_indexed = True
        else:
            merged[fn] = DocumentInfo(
                filename=fn,
                vector_rag_indexed=False,
                bm25_indexed=True,
                chunks=0,
            )

    return list(merged.values())


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Delete a document from both RAG systems.

    Toggleable via the 'document_deletion' feature flag in lua/features.lua.
    """
    if not is_enabled("document_deletion"):
        raise HTTPException(
            status_code=403,
            detail="Document deletion is disabled (set document_deletion.enabled = true in lua/features.lua)",
        )

    deleted_any = False

    # Remove from Vector RAG (Qdrant)
    try:
        vr_deleted = vector_rag.delete_document(filename)
        if vr_deleted:
            deleted_any = True
    except Exception as e:
        if is_enabled("debug_logging"):
            print(f"[main] vector_rag.delete_document failed: {e}")

    # Remove from BM25 index
    try:
        bm_deleted = bm25_rag.delete_document(filename)
        if bm_deleted:
            deleted_any = True
    except Exception as e:
        if is_enabled("debug_logging"):
            print(f"[main] bm25_rag.delete_document failed: {e}")

    if not deleted_any:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in any index",
        )

    return {
        "status": "success",
        "document": filename,
        "message": f"Deleted from Vector RAG and BM25 index",
    }
