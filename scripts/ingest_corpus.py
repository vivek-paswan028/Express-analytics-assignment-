#!/usr/bin/env python3
"""
scripts/ingest_corpus.py
─────────────────────────
One-shot script to ingest all corpus documents into ChromaDB.

Useful for:
- Re-ingesting after deleting the chroma_db directory.
- Verifying the ingestion pipeline before starting the API.
- CI/CD pre-population of the vector store.

Usage:
    cd rag-docs-assistant
    cp .env.example .env   # add your GOOGLE_API_KEY
    python scripts/ingest_corpus.py

Note: The API server also auto-ingests on startup if the collection is empty.
This script is for standalone ingestion without starting the API.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from ingestion.chunker import chunk_documents
from ingestion.loader import load_corpus
from ingestion.vector_store import init_vector_store

settings = get_settings()
CORPUS_DIR = Path(__file__).parent.parent / "docs" / "corpus"


def main():
    print("RAG Documentation Assistant — Corpus Ingestion")
    print("=" * 50)
    print(f"Corpus directory: {CORPUS_DIR}")
    print(f"Vector store:     {settings.chroma_persist_dir}")
    print(f"Chunk size:       {settings.chunk_size} chars")
    print(f"Chunk overlap:    {settings.chunk_overlap} chars")
    print()

    # Load documents
    docs = load_corpus(CORPUS_DIR)
    if not docs:
        print("ERROR: No documents found in corpus directory.")
        print(f"Run scripts/fetch_corpus.py first, or check {CORPUS_DIR}")
        sys.exit(1)

    print(f"Loaded {len(docs)} document(s):")
    for doc in docs:
        print(f"  - {doc.metadata.get('source', '?')} ({len(doc.page_content)} chars)")

    # Chunk
    print()
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and index
    print()
    print("Initialising vector store (embedding model may download on first run)...")
    store = init_vector_store(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_model_name=settings.embedding_model,
    )

    result = store.add_documents(chunks)
    print()
    print("Ingestion complete:")
    print(f"  Chunks added:              {result['added']}")
    print(f"  Chunks skipped (duplicate): {result['skipped_duplicate']}")
    print(f"  Total chunks in store:     {store.get_total_chunks()}")
    print()

    # Verify with a test query
    print("Verification — test retrieval query: 'conditional edge LangGraph'")
    test_results = store.similarity_search("conditional edge LangGraph", k=3)
    for i, doc in enumerate(test_results, 1):
        source = doc.metadata.get("source", "?")
        heading = doc.metadata.get("heading", "")
        score = doc.metadata.get("score", "?")
        print(f"  {i}. {source} — {heading} (score={score})")

    print()
    print("Ingestion successful. You can now start the API server:")
    print("  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()

