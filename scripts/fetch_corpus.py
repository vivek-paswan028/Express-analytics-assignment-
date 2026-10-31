#!/usr/bin/env python3
"""
scripts/fetch_corpus.py
────────────────────────
Reproducible corpus fetch script.

Run this to (re-)download the corpus documents from their canonical
GitHub raw URLs into docs/corpus/. The documents are also committed
to the repository, so this script is only needed if you want to
refresh them or start fresh.

Usage:
    cd rag-docs-assistant
    python scripts/fetch_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

CORPUS_DIR = Path(__file__).parent.parent / "docs" / "corpus"

# Canonical URLs for the 5 corpus documents
CORPUS_SOURCES = [
    {
        "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/low_level.md",
        "filename": "langgraph_concepts.md",
        "fallback": None,  # use local file if download fails
    },
    {
        "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/concepts/rag.mdx",
        "filename": "langchain_rag.md",
        "fallback": None,
    },
    {
        "url": "https://raw.githubusercontent.com/chroma-core/chroma/main/README.md",
        "filename": "chromadb_guide.md",
        "fallback": None,
    },
    {
        "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
        "filename": "fastapi_guide.md",
        "fallback": None,
    },
    {
        "url": "https://raw.githubusercontent.com/pydantic/pydantic/main/docs/concepts/models.md",
        "filename": "pydantic_v2.md",
        "fallback": None,
    },
]


def fetch_corpus():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "RAG-Docs-Assistant/1.0"}

    for source in CORPUS_SOURCES:
        dest = CORPUS_DIR / source["filename"]
        print(f"Fetching {source['filename']} ...", end=" ")

        try:
            resp = requests.get(source["url"], headers=headers, timeout=30)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
            print(f"OK ({len(resp.text)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
            if dest.exists():
                print(f"  → Using existing local file: {dest.name}")
            else:
                print(f"  → No local fallback. Using bundled document.")

    print(f"\nCorpus ready in: {CORPUS_DIR}")
    print(f"Documents: {len(list(CORPUS_DIR.glob('*.md')))}")


if __name__ == "__main__":
    fetch_corpus()

