from __future__ import annotations

import logging

from langchain_core.documents import Document
from ingestion.vector_store import get_vector_store
from rag.state import RAGState

logger = logging.getLogger(__name__)


def retrieval_node(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["original_question"]
    original_q = state.get("original_question", "")
    top_k = state.get("top_k", 5)

    try:
        store = get_vector_store()
        docs = store.similarity_search(query=query, k=top_k)

        combined_text = (original_q + " " + query).lower()
        indexed_docs = store.list_documents()

        matching_doc_ids = set()
        is_resume_query = any(term in combined_text for term in ["resume", "cv", "candidate", "applicant", "profile", "vivek"])

        for idoc in indexed_docs:
            src = idoc.get("source", "").lower()
            did = idoc.get("doc_id", "")
            if is_resume_query and any(r in src for r in ["resume", "rsm", "cv", "vivek"]):
                matching_doc_ids.add(did)
            base_src = src.split(".")[0].lower()
            if base_src and len(base_src) > 3 and base_src in combined_text:
                matching_doc_ids.add(did)

        if matching_doc_ids:
            for mid in matching_doc_ids:
                doc_chunks = store.get_chunks_by_document(mid)
                existing_ids = {d.metadata.get("chunk_id") for d in docs}
                for c in doc_chunks:
                    if c.metadata.get("chunk_id") not in existing_ids:
                        docs.append(c)

        logger.info(
            "Retrieval: '%s' → %d chunks (top_k=%d)",
            query[:80], len(docs), top_k,
        )
        return {"retrieved_documents": docs}

    except Exception as e:
        logger.error("retrieval_node failed: %s", e)
        return {
            "retrieved_documents": [],
            "error": f"Retrieval failed: {str(e)}",
        }

