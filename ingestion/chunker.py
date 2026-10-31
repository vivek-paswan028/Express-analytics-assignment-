from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _detect_nearest_heading(text: str, full_doc: str, char_start: int) -> str:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
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

            enriched_metadata = {
                **chunk.metadata,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "char_start": char_start,
                "heading": heading,
                "content_hash": _content_hash(chunk.page_content),
                "chunk_size_actual": len(chunk.page_content),
            }
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

