"""
ingestion/chunker.py
─────────────────────
Split LangChain Documents into smaller chunks suitable for embedding.

Chunking strategy for technical documentation:
- chunk_size=800 chars: large enough to keep code examples intact,
  small enough to remain semantically focused for embedding.
- chunk_overlap=100 chars: prevents concept truncation at boundaries.
- Separators ordered from largest semantic unit to smallest:
  Markdown headings first, then paragraphs, then lines, then words.
  This keeps related prose together and avoids cutting inside code blocks.

A per-chunk content_hash (SHA-256 prefix) enables deduplication:
when re-ingesting a document that hasn't changed, duplicate chunks
are detected and skipped rather than stored twice.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _detect_nearest_heading(text: str, full_doc: str, char_start: int) -> str:
    """
    Find the nearest Markdown heading at or before char_start in full_doc.
    Returns the heading text without the # prefix, or "" if none found.
    """
    preceding = full_doc[:char_start]
    headings = re.findall(r"^#{1,4}\s+(.+)$", preceding, re.MULTILINE)
    return headings[-1].strip() if headings else ""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split a list of Documents into chunks.

    Each chunk carries extended metadata:
        chunk_id     — unique identifier: {doc_id}_chunk_{index:04d}
        chunk_index  — 0-based index within the document
        char_start   — character offset in the original document
        heading      — nearest Markdown heading above the chunk
        content_hash — SHA-256 prefix for deduplication
        chunk_size_actual — actual character length of this chunk

    Args:
        documents:     LangChain Document objects (one per source file).
        chunk_size:    Target maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # For Markdown: try to split at headings first, then paragraphs,
        # then lines, then words. This keeps sections together.
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,   # adds "start_index" to metadata
    )

    all_chunks: list[Document] = []
    chunk_counters: dict[str, int] = {}

    for doc in documents:
        raw_chunks = splitter.split_documents([doc])

        doc_id = doc.metadata.get("doc_id", "unknown")
        full_text = doc.page_content

        for chunk in raw_chunks:
            idx = chunk_counters.get(doc_id, 0)
            chunk_counters[doc_id] = idx + 1

            char_start = chunk.metadata.get("start_index", 0)
            heading = _detect_nearest_heading(chunk.page_content, full_text, char_start)
            chunk_id = f"{doc_id}_chunk_{idx:04d}"

            # Build enriched metadata (preserve all original metadata)
            enriched_metadata = {
                **chunk.metadata,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "char_start": char_start,
                "heading": heading,
                "content_hash": _content_hash(chunk.page_content),
                "chunk_size_actual": len(chunk.page_content),
            }
            # Remove start_index (renamed to char_start)
            enriched_metadata.pop("start_index", None)

            all_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=enriched_metadata,
                )
            )

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks

