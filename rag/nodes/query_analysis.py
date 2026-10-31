"""
rag/nodes/query_analysis.py
────────────────────────────
Node 1 of the LangGraph RAG workflow.

Responsibilities:
1. On the FIRST pass (retry_count == 0):
   - Classify the question type (conceptual, how-to, troubleshooting, api-reference).
   - Rewrite/expand the query to improve retrieval quality.

2. On RETRY passes (retry_count > 0):
   - Produce a substantially different reformulation of the query.
   - Use the retry number as a hint to try a broader or narrower angle.
   - Increment retry_count in state.

Structured output via Pydantic ensures the LLM returns valid JSON
rather than free text, avoiding parsing failures.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import get_settings
from rag.llm import get_gemini_llm
from rag.state import RAGState

logger = logging.getLogger(__name__)

settings = get_settings()


class QueryAnalysisResult(BaseModel):
    """Structured output from the query analysis LLM call."""

    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference", "conversational", "unknown"] = Field(
        description="Classification of the user's question"
    )
    rewritten_query: str = Field(
        description=(
            "An improved version of the question optimised for vector similarity search. "
            "Use specific technical terminology, expand acronyms, and remove ambiguity. "
            "Return only the query string."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining why you chose this rewrite."
    )


def _build_llm() -> ChatGoogleGenerativeAI:
    return get_gemini_llm(temperature=settings.llm_temperature)


def query_analysis_node(state: RAGState) -> dict:
    """
    Analyse and rewrite the query for better retrieval.

    On retry passes this node additionally increments retry_count,
    and crafts a substantially different reformulation to surface
    different chunks from the vector store.
    """
    question = state["original_question"]
    retry_count = state.get("retry_count", 0)
    is_retry = retry_count > 0

    # Fast path for conversational greetings
    normalized = question.strip().lower()
    greetings = {"hi", "hii", "hello", "hey", "heyy", "good morning", "good evening", "thanks", "thank you", "help"}
    if normalized in greetings:
        logger.info("Fast path: detected conversational greeting '%s'", normalized)
        return {
            "query_type": "conversational",
            "rewritten_query": question,
        }

    try:
        llm = _build_llm()
        structured_llm = llm.with_structured_output(QueryAnalysisResult)

        if is_retry:
            # Retry prompt: ask for a different angle than the previous attempt
            previous_query = state.get("rewritten_query", question)
            system_msg = SystemMessage(content=(
                "You are improving a search query for a technical documentation retrieval system. "
                "The previous query did not find relevant results. "
                "Generate a substantially DIFFERENT reformulation that:\n"
                "- Uses alternative technical terminology or synonyms\n"
                "- Tries a broader or more specific angle\n"
                "- Avoids the same phrasing as the previous query\n"
                "- Focuses on the underlying technical concept\n"
                "Classify the query type and provide a new rewritten_query."
            ))
            human_msg = HumanMessage(content=(
                f"Original question: {question}\n"
                f"Previous query (did not work): {previous_query}\n"
                f"Retry attempt: {retry_count}\n\n"
                "Generate a different search query."
            ))
        else:
            # First pass: classify and rewrite
            system_msg = SystemMessage(content=(
                "You are a query analyser for a technical documentation assistant. "
                "Given a user question, you must:\n"
                "1. Classify it as: conceptual (what/why questions), how-to (step-by-step), "
                "troubleshooting (fix errors/issues), api-reference (function params/types), "
                "or unknown.\n"
                "2. Rewrite it as a short, precise search query that:\n"
                "   - Expands acronyms (e.g. LLM → large language model)\n"
                "   - Adds relevant synonyms\n"
                "   - Removes conversational filler ('can you', 'please', 'how do I')\n"
                "   - Retains the technical core concept\n"
                "Return structured output only."
            ))
            human_msg = HumanMessage(content=f"Question: {question}")

        result: QueryAnalysisResult = structured_llm.invoke([system_msg, human_msg])

        logger.info(
            "Query analysis (retry=%d): type=%s | '%s' → '%s'",
            retry_count, result.query_type, question[:60], result.rewritten_query[:80],
        )

        update: dict = {
            "rewritten_query": result.rewritten_query,
            "query_type": result.query_type,
        }
        # Increment retry count on retries
        if is_retry:
            update["retry_count"] = retry_count + 1

        return update

    except Exception as e:
        logger.error("query_analysis_node failed: %s", e)
        # Fallback: use original question as-is; don't crash the graph
        update = {
            "rewritten_query": question,
            "query_type": "unknown",
        }
        if is_retry:
            update["retry_count"] = retry_count + 1
        return update

