from __future__ import annotations

import logging

from rag.state import RAGState

logger = logging.getLogger(__name__)


def route_after_query_analysis(state: RAGState) -> str:
    if state.get("query_type") == "greeting":
        logger.info("Routing after analysis: greeting detected → fallback")
        return "fallback"
    return "retrieval"


def route_after_grading(state: RAGState) -> str:
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

    if state.get("query_type") == "greeting":
        logger.info("Routing: pure greeting query → fallback")
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

