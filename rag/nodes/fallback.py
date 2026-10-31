from __future__ import annotations

import logging

from rag.state import RAGState

logger = logging.getLogger(__name__)

_NO_CONTEXT_MESSAGE = (
    "I could not find sufficient information in the indexed documentation "
    "to answer this question.\n\n"
    "This may mean:\n"
    "- The topic is not covered by the currently indexed documents.\n"
    "- The question uses terminology that doesn't match the documentation.\n"
    "- Relevant documents have not been ingested yet.\n\n"
    "You can ingest additional documents via POST /ingest, "
    "or check what is indexed via GET /documents."
)

_SYSTEM_ERROR_MESSAGE = (
    "I encountered a system error while processing your question and could not "
    "generate a reliable answer.\n\n"
    "Error details: {error}\n\n"
    "Please check your API key configuration and try again."
)


_GREETING_MESSAGE = (
    "Hello! 👋 I am your Technical Documentation Assistant.\n\n"
    "I am grounded in the technical documentation you index. "
    "You can ask me questions about your documentation (e.g. LangGraph, FastAPI, ChromaDB), "
    "or upload new guides using the **📁 Manage & Ingest Documents** tab in the sidebar!"
)


def fallback_node(state: RAGState) -> dict:
    error = state.get("error")
    retry_count = state.get("retry_count", 0)

    if state.get("query_type") == "conversational":
        return {
            "answer": _GREETING_MESSAGE,
            "sources": [],
            "generation_supported": True,
        }

    if error:
        logger.warning(
            "Fallback triggered by system error (retry=%d): %s",
            retry_count, error,
        )
        user_error = error.split(":")[0] if ":" in error else "Internal error"
        answer = _SYSTEM_ERROR_MESSAGE.format(error=user_error)
    else:
        logger.info(
            "Fallback: no relevant documents found after %d retries for: '%s'",
            retry_count, state.get("original_question", "")[:80],
        )
        answer = _NO_CONTEXT_MESSAGE

    return {
        "answer": answer,
        "sources": [],
        "generation_supported": False,
    }

