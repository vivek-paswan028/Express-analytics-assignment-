from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self, persist_dir: str, collection_name: str, embedding_model_name: str):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embedding_model = get_embedding_model(embedding_model_name)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready: collection='%s', path='%s', %d chunks indexed",
            collection_name, persist_dir, self._collection.count(),
        )


    def add_documents(self, chunks: list[Document]) -> dict[str, int]:
        if not chunks:
            return {"added": 0, "skipped_duplicate": 0}

        existing_hashes: set[str] = set()
        try:
            existing = self._collection.get(include=["metadatas"])
            for meta in (existing["metadatas"] or []):
                if meta and "content_hash" in meta:
                    existing_hashes.add(meta["content_hash"])
        except Exception as e:
            logger.warning("Could not fetch existing hashes: %s", e)

        to_add_ids: list[str] = []
        to_add_docs: list[str] = []
        to_add_metas: list[dict] = []
        seen_batch_ids: set[str] = set()
        skipped = 0
        ingested_at = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            content_hash = chunk.metadata.get("content_hash", "")
            if content_hash and content_hash in existing_hashes:
                skipped += 1
                continue
            if content_hash:
                existing_hashes.add(content_hash)

            base_chunk_id = chunk.metadata.get("chunk_id", f"chunk_{len(to_add_ids)}")
            chunk_id = base_chunk_id
            suffix = 1
            while chunk_id in seen_batch_ids:
                chunk_id = f"{base_chunk_id}_{suffix}"
                suffix += 1

            seen_batch_ids.add(chunk_id)
            meta = {k: str(v) for k, v in chunk.metadata.items()}
            meta["chunk_id"] = chunk_id
            meta["ingested_at"] = ingested_at

            to_add_ids.append(chunk_id)
            to_add_docs.append(chunk.page_content)
            to_add_metas.append(meta)

        if not to_add_ids:
            logger.info("All %d chunks were duplicates; nothing added.", skipped)
            return {"added": 0, "skipped_duplicate": skipped}

        embeddings = self._embedding_model.embed_documents(to_add_docs)

        BATCH_SIZE = 500
        for i in range(0, len(to_add_ids), BATCH_SIZE):
            self._collection.upsert(
                ids=to_add_ids[i:i + BATCH_SIZE],
                documents=to_add_docs[i:i + BATCH_SIZE],
                embeddings=embeddings[i:i + BATCH_SIZE],
                metadatas=to_add_metas[i:i + BATCH_SIZE],
            )

        logger.info("Added %d chunks (%d skipped as duplicates).", len(to_add_ids), skipped)
        return {"added": len(to_add_ids), "skipped_duplicate": skipped}


    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        if self._collection.count() == 0:
            logger.warning("Vector store is empty — no documents indexed.")
            return []

        query_embedding = self._embedding_model.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, dist in zip(documents, metadatas, distances):
            meta_copy = dict(meta or {})
            meta_copy["score"] = round(float(dist), 4)
            docs.append(Document(page_content=text, metadata=meta_copy))

        return docs


    def list_documents(self) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []

        all_items = self._collection.get(include=["metadatas"])
        metadatas = all_items.get("metadatas") or []

        docs: dict[str, dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "chunks": 0,
                    "ingested_at": meta.get("ingested_at", ""),
                }
            docs[doc_id]["chunks"] += 1

        return sorted(docs.values(), key=lambda d: d["doc_id"])

    def get_total_chunks(self) -> int:
        return self._collection.count()

    def document_exists(self, doc_id: str) -> bool:
        result = self._collection.get(where={"doc_id": doc_id}, limit=1)
        return bool(result.get("ids"))

    def clear_all_documents(self) -> int:
        count = self._collection.count()
        if count > 0:
            all_items = self._collection.get()
            all_ids = all_items.get("ids", [])
            if all_ids:
                self._collection.delete(ids=all_ids)
        logger.info("Cleared all %d documents from collection '%s'", count, self._collection_name)
        return count

    def delete_document(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)


_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store_instance
    if _store_instance is None:
        from app.config import get_settings
        settings = get_settings()
        logger.info("Auto-initialising VectorStore singleton on demand...")
        _store_instance = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            embedding_model_name=settings.embedding_model,
        )
    return _store_instance


def init_vector_store(
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
) -> VectorStore:
    global _store_instance
    _store_instance = VectorStore(persist_dir, collection_name, embedding_model_name)
    return _store_instance

