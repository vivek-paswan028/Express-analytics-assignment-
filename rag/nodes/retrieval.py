from __future__ import annotations

import logging

from ingestion.vector_store import get_vector_store
from rag.state import RAGState

logger = logging.getLogger(__name__)


def retrieval_node(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["original_question"]
    top_k = state.get("top_k", 5)

    try:
        store = get_vector_store()
        docs = store.similarity_search(query=query, k=top_k)

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

