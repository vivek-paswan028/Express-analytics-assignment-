from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from rag.llm import get_gemini_llm
from rag.state import RAGState, SourceMetadata

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

_GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are an expert AI and software engineering assistant.
The user asked a technical question that was not found in their indexed documentation corpus.
Provide a comprehensive, accurate, and well-structured answer based on your general technical knowledge.

Format requirements:
1. Begin with a clear notice block:
> ℹ️ **General Technical Knowledge Response**  
> *This topic was not found in your indexed documentation. The following answer is provided from general technical knowledge:*

2. Provide a thorough, complete, and well-structured explanation using Markdown headings, bullet points, and code examples if applicable.
"""


def fallback_node(state: RAGState) -> dict:
    error = state.get("error")
    retry_count = state.get("retry_count", 0)

    if state.get("query_type") == "greeting":
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
        return {
            "answer": answer,
            "sources": [],
            "generation_supported": False,
        }

    question = state.get("original_question", "")
    logger.info(
        "Fallback: generating answer using general LLM knowledge for: '%s'",
        question[:80],
    )
    try:
        llm = get_gemini_llm(temperature=0.3)
        res = llm.invoke([
            SystemMessage(content=_GENERAL_KNOWLEDGE_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}"),
        ])
        raw_content = res.content
        if isinstance(raw_content, str):
            answer = raw_content.strip()
        elif isinstance(raw_content, list):
            parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content]
            answer = "".join(parts).strip()
        else:
            answer = str(raw_content).strip()

        sources = [
            SourceMetadata(
                doc_id="general_knowledge",
                source="General AI Knowledge (Not in Corpus)",
                chunk_id="",
                heading="General Knowledge Fallback",
            )
        ]
        return {
            "answer": answer,
            "sources": sources,
            "generation_supported": True,
        }
    except Exception as e:
        logger.error("Fallback general knowledge generation failed: %s", e)
        return {
            "answer": _NO_CONTEXT_MESSAGE,
            "sources": [],
            "generation_supported": False,
        }

