"""
rag/nodes/grading.py
─────────────────────
Node 3 of the LangGraph RAG workflow — the self-corrective component.

Each retrieved chunk is graded by an LLM as "relevant" or "irrelevant"
to the original question. Only relevant chunks proceed to generation.

Design decisions:
- Structured output with Literal["relevant", "irrelevant"] prevents
  free-text grades that need regex parsing (which can silently fail).
- We grade against original_question (not rewritten_query) because
  the user cares about the original intent; the rewrite is just a
  retrieval aid.
- Conservative default: if the LLM returns an unexpected response or
  raises an exception for a chunk, we grade it "irrelevant" to avoid
  passing hallucination-prone context to the generation node.
- We run grading calls sequentially (not async) to stay within free-tier
  rate limits. For production, switch to asyncio.gather with rate limiting.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import get_settings
from rag.state import RAGState

logger = logging.getLogger(__name__)

settings = get_settings()

_GRADE_SYSTEM_PROMPT = (
    "You are evaluating whether a document chunk is relevant to a user's question.\n\n"
    "Grade as 'relevant' if the chunk:\n"
    "  - Directly answers or contributes to answering the question\n"
    "  - Contains definitions, examples, or explanations of the concept asked about\n"
    "  - Provides code or API details the question asks about\n\n"
    "Grade as 'irrelevant' if the chunk:\n"
    "  - Is about a completely different topic\n"
    "  - Contains only tangential mentions of keywords without substantive content\n"
    "  - Is boilerplate (license text, table of contents entries, etc.)\n\n"
    "Be conservative: only grade 'relevant' when there is clear topical overlap."
)


class DocumentGrade(BaseModel):
    """Structured grade for a single retrieved document chunk."""

    score: Literal["relevant", "irrelevant"] = Field(
        description="'relevant' if the chunk helps answer the question, 'irrelevant' otherwise."
    )
    reasoning: str = Field(
        description="One sentence explaining the grade."
    )


def _grade_single_document(
    llm_with_output: object,
    question: str,
    doc: Document,
) -> bool:
    """
    Grade one document. Returns True if relevant, False if irrelevant.
    Catches exceptions and defaults to irrelevant (conservative).
    """
    source = doc.metadata.get("source", "unknown")
    heading = doc.metadata.get("heading", "")
    location = f"{source} — {heading}" if heading else source

    human_content = (
        f"Question: {question}\n\n"
        f"Document chunk (from {location}):\n"
        f"{doc.page_content[:1200]}\n\n"   # cap to avoid token overflow
        "Is this chunk relevant to the question?"
    )

    try:
        grade: DocumentGrade = llm_with_output.invoke(
            [SystemMessage(content=_GRADE_SYSTEM_PROMPT),
             HumanMessage(content=human_content)]
        )
        is_relevant = grade.score == "relevant"
        logger.debug(
            "Grade: %s | chunk=%s | reason: %s",
            grade.score.upper(), doc.metadata.get("chunk_id", "?"), grade.reasoning[:80],
        )
        return is_relevant

    except Exception as e:
        logger.warning(
            "Grading failed for chunk %s: %s — defaulting to irrelevant",
            doc.metadata.get("chunk_id", "?"), e,
        )
        return False  # conservative default


def grading_node(state: RAGState) -> dict:
    """
    Grade all retrieved documents and filter to only relevant ones.

    Returns an update to state with:
    - relevant_documents: filtered list of relevant chunks
    """
    question = state["original_question"]
    retrieved = state.get("retrieved_documents", [])

    if not retrieved:
        logger.info("Grading: no documents to grade (retrieval returned 0 results).")
        return {"relevant_documents": []}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.0,  # grading must be deterministic
            google_api_key=settings.google_api_key,
        )
        grader = llm.with_structured_output(DocumentGrade)

        relevant: list[Document] = []
        for doc in retrieved:
            if _grade_single_document(grader, question, doc):
                relevant.append(doc)

        logger.info(
            "Grading complete: %d/%d chunks are relevant.",
            len(relevant), len(retrieved),
        )
        return {"relevant_documents": relevant}

    except Exception as e:
        logger.error("grading_node encountered a top-level error: %s", e)
        # If grading itself fails entirely, pass all retrieved docs to generation
        # rather than sending to fallback — it's better to potentially answer
        # than to silently fail on a grading API error.
        return {
            "relevant_documents": retrieved,
            "error": f"Grading partially failed: {str(e)}",
        }

