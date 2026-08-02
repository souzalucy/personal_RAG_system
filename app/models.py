from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class Source(BaseModel):
    method: str  # "vector_rag" or "bm25"
    content: str
    document: str
    page: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    status: str
    document: str
    vector_rag: str
    bm25: str
    chunks: int = 0


class FolderIngestResponse(BaseModel):
    status: str
    folder: str
    total_pdfs: int
    succeeded: int
    failed: int
    results: list[IngestResponse]


class DocumentInfo(BaseModel):
    filename: str
    vector_rag_indexed: bool
    bm25_indexed: bool
    chunks: int = 0
