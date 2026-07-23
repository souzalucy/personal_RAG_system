#!/usr/bin/env python3
"""CLI script to ingest PDFs into both RAG systems.

Usage:
    python scripts/ingest.py <path-to-pdf-or-folder>
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import vector_rag, bm25_rag


def process_pdf(file_path: str) -> None:
    """Process a single PDF through both RAG pipelines."""
    filename = os.path.basename(file_path)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    # --- Vector RAG ---
    print("\n[1/2] Vector RAG (docling → Qdrant)...")
    try:
        chunk_count = vector_rag.index_document(file_path)
        print(f"  ✓ Indexed {chunk_count} chunks into Qdrant")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # --- BM25 (vectorless) ---
    print("\n[2/2] BM25 (keyword-based, vectorless RAG)...")
    try:
        _, page_texts = vector_rag.parse_document(file_path)
        page_count = bm25_rag.index_document(file_path, [p["text"] for p in page_texts])
        print(f"  ✓ Indexed {page_count} pages into BM25 index")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    path = Path(target)

    if not path.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            print(f"Error: {path} is not a PDF file")
            sys.exit(1)
        process_pdf(str(path))
    elif path.is_dir():
        pdfs = sorted(path.glob("*.pdf")) + sorted(path.glob("*.PDF"))
        if not pdfs:
            print(f"No PDF files found in {target}")
            sys.exit(1)
        print(f"Found {len(pdfs)} PDF(s) in {target}")
        for pdf in pdfs:
            process_pdf(str(pdf))
    else:
        print(f"Error: {target} is neither a file nor a directory")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
