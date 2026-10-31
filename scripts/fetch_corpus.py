from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

CORPUS_DIR = Path(__file__).parent.parent / "docs" / "corpus"

CORPUS_SOURCES = [
    {
        "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/docs/concepts/low_level.md",
        "filename": "langgraph_concepts.md",
        "fallback": None,
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

