"""
rag/nodes/retrieval.py
───────────────────────
Node 2 of the LangGraph RAG workflow.

Performs vector similarity search using the current rewritten_query.
Returns top-k chunks with full metadata preserved.

If the vector store is empty or raises an exception, the node returns
an empty list rather than crashing — the grading node handles zero results
by routing to the retry/fallback path.
"""
from __future__ import annotations

import logging

from ingestion.vector_store import get_vector_store
from rag.state import RAGState

logger = logging.getLogger(__name__)


def retrieval_node(state: RAGState) -> dict:
    """
    Search the vector store for chunks relevant to the current query.

    Uses state["rewritten_query"] (not original_question) so that
    query_analysis rewrites actually affect retrieval.
    """
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

