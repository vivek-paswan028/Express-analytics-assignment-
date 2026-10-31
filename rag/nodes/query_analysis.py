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

    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference", "conversational", "greeting", "unknown"] = Field(
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
    question = state["original_question"]
    retry_count = state.get("retry_count", 0)
    is_retry = bool(state.get("retrieved_documents")) or retry_count > 0

    normalized = question.strip().lower().rstrip("!?.")
    greetings = {
        "hi", "hii", "hiii", "hello", "hey", "heyy", "heyyy",
        "good morning", "good evening", "good afternoon",
        "howdy", "sup", "yo", "greeting", "greetings"
    }
    tokens = normalized.split()
    non_greeting_words = {"resume", "candidate", "project", "skill", "upload", "uploaded", "document", "docs", "file", "fastapi", "langgraph", "chroma", "pydantic", "langchain", "name"}
    is_greeting = (
        normalized in greetings
        or (len(tokens) <= 2 and any(t in greetings for t in tokens) and not any(t in non_greeting_words for t in tokens))
    )
    if is_greeting:
        logger.info("Fast path: detected pure greeting '%s'", normalized)
        return {
            "query_type": "greeting",
            "rewritten_query": question,
        }

    try:
        llm = _build_llm()
        structured_llm = llm.with_structured_output(QueryAnalysisResult)

        if is_retry:
            previous_query = state.get("rewritten_query", question)
            system_msg = SystemMessage(content=(
                "You are improving a search query for a technical documentation retrieval system. "
                "The previous query did not find relevant results in the indexed documents. "
                "Generate a substantially DIFFERENT reformulation with alternative terminology, broader concepts, or synonyms. "
                "Classify the query type and provide a new rewritten_query."
            ))
            human_msg = HumanMessage(content=(
                f"Original question: {question}\n"
                f"Previous query (did not work): {previous_query}\n"
                f"Retry attempt: {retry_count}\n\n"
                "Generate a different search query."
            ))
        else:
            system_msg = SystemMessage(content=(
                "You are a query analyser for a technical documentation and document assistant. "
                "Given a user question, you must:\n"
                "1. Classify it as: conceptual, how-to, troubleshooting, api-reference, conversational, or unknown.\n"
                "2. Rewrite it as an effective search query for vector retrieval:\n"
                "   - If asking about a resume or candidate: rewrite to include 'candidate name contact summary technical skills projects education experience'\n"
                "   - Expands acronyms (e.g. RAG -> retrieval-augmented generation)\n"
                "   - Adds relevant technical synonyms and removes conversational filler\n"
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
        if is_retry:
            update["retry_count"] = retry_count + 1

        return update

    except Exception as e:
        logger.error("query_analysis_node failed: %s", e)
        update = {
            "rewritten_query": question,
            "query_type": "unknown",
        }
        if is_retry:
            update["retry_count"] = retry_count + 1
        return update

