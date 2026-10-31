"""
rag/routing.py
───────────────
Conditional edge routing functions for the LangGraph workflow.

Routing functions are pure functions of state — they read state fields
and return string node names. They contain no side effects.

Graph routing logic:
  After query_analysis → always go to retrieval
  After retrieval       → always go to grading
  After grading         → route_after_grading():
      - relevant docs found          → "generate"
      - no relevant + retries left   → "query_analysis"  (rewrite + re-retrieve)
      - no relevant + retries spent  → "fallback"
      - error in state               → "fallback"
"""
from __future__ import annotations

import logging

from rag.state import RAGState

logger = logging.getLogger(__name__)


def route_after_query_analysis(state: RAGState) -> str:
    """
    Decide what to do immediately after query analysis.

    If the question is recognized as a conversational greeting or meta-question,
    skip retrieval and grading entirely and route straight to fallback (which provides
    the assistant greeting). Otherwise proceed to vector retrieval.
    """
    if state.get("query_type") == "conversational":
        logger.info("Routing after analysis: conversational greeting detected → fallback")
        return "fallback"
    return "retrieval"


def route_after_grading(state: RAGState) -> str:
    """
    Decide what to do after document grading.

    Returns one of: "generate" | "query_analysis" | "fallback"

    Priority of checks:
    1. If state["error"] is set → fallback (something broke upstream)
    2. If relevant_documents is non-empty → generate
    3. If retry_count < max_retries → query_analysis (will increment count)
    4. Otherwise → fallback (retries exhausted)
    """
    error = state.get("error")
    if error:
        logger.info("Routing: error detected → fallback")
        return "fallback"

    relevant = state.get("relevant_documents", [])
    if relevant:
        logger.info(
            "Routing: %d relevant documents found → generate",
            len(relevant),
        )
        return "generate"

    # Conversational greetings skip retry cycles
    if state.get("query_type") == "conversational":
        logger.info("Routing: conversational query → fallback")
        return "fallback"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        logger.info(
            "Routing: no relevant docs, retry %d/%d → query_analysis",
            retry_count + 1, max_retries,
        )
        return "query_analysis"

    logger.info(
        "Routing: no relevant docs, retries exhausted (%d/%d) → fallback",
        retry_count, max_retries,
    )
    return "fallback"

