"""
rag/nodes/fallback.py
──────────────────────
Terminal fallback node — reached when:
  1. retry_count >= max_retries AND no relevant documents found, OR
  2. A node sets state["error"] (e.g. LLM API failure).

Returns an honest "I don't know" response rather than a hallucinated answer.
The response distinguishes between "topic not in corpus" and "system error"
so the user can take appropriate action.
"""
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


def fallback_node(state: RAGState) -> dict:
    """
    Return a structured "I don't know" or error response.

    Distinguishes between two scenarios:
    - error in state → system failure message
    - no error → simply no relevant context found after retries
    """
    error = state.get("error")
    retry_count = state.get("retry_count", 0)

    if error:
        # System error (LLM/retrieval failure)
        logger.warning(
            "Fallback triggered by system error (retry=%d): %s",
            retry_count, error,
        )
        # Provide a user-facing message without exposing raw stack traces
        user_error = error.split(":")[0] if ":" in error else "Internal error"
        answer = _SYSTEM_ERROR_MESSAGE.format(error=user_error)
    else:
        # Exhausted retries with no relevant context
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

