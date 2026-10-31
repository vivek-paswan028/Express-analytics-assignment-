"""
rag/nodes/grading.py
─────────────────────
Node 3 of the LangGraph RAG workflow — the self-corrective component.

Evaluates whether retrieved chunks are relevant to the user's question.
High-Performance Batch Grading:
Instead of making N sequential API roundtrips (1 per chunk), this node
grades all retrieved chunks in a single structured LLM call. This reduces
latency from 15-30s down to 1-2s and avoids free-tier rate limit exhaustion.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from rag.llm import get_gemini_llm
from rag.state import RAGState

logger = logging.getLogger(__name__)

settings = get_settings()

_BATCH_GRADE_SYSTEM_PROMPT = """You are a strict technical relevance grader.
Given a user question and a list of numbered document chunks, evaluate each chunk independently.

Grading criteria:
- 'relevant': The chunk contains information, definitions, code, or context that directly helps answer or understand the question.
- 'irrelevant': The chunk is about a different topic, only contains tangential keyword hits, or is pure boilerplate.

Return a structured evaluation for every chunk index provided."""


class ChunkEvaluation(BaseModel):
    chunk_index: int = Field(description="The 0-based index of the chunk ([Chunk 0], [Chunk 1], etc.)")
    score: Literal["relevant", "irrelevant"] = Field(description="'relevant' or 'irrelevant'")
    reasoning: str = Field(description="One brief sentence explaining the grade.")


class BatchGradingResult(BaseModel):
    evaluations: list[ChunkEvaluation] = Field(description="List of evaluations for all provided chunks.")


# Fallback single-chunk grade model (for unit test compatibility)
class DocumentGrade(BaseModel):
    score: Literal["relevant", "irrelevant"] = Field(description="'relevant' or 'irrelevant'")
    reasoning: str = Field(description="One brief sentence explaining the grade.")


def grading_node(state: RAGState) -> dict:
    """
    Grade all retrieved documents in a single structured batch call.

    Returns an update to state with:
    - relevant_documents: filtered list of chunks graded as 'relevant'
    """
    question = state["original_question"]
    retrieved = state.get("retrieved_documents", [])

    if not retrieved:
        logger.info("Grading: no documents to grade (retrieval returned 0 results).")
        return {"relevant_documents": []}

    # If question is classified as conversational, skip grading
    if state.get("query_type") == "conversational":
        return {"relevant_documents": []}

    try:
        llm = get_gemini_llm(temperature=0.0)

        # Build combined prompt with all chunks numbered
        chunk_descriptions = []
        for i, doc in enumerate(retrieved):
            source = doc.metadata.get("source", "unknown")
            heading = doc.metadata.get("heading", "")
            location = f"{source} — {heading}" if heading else source
            # Trim chunk preview to 500 chars to avoid token bloat and latency
            chunk_descriptions.append(
                f"[Chunk {i}] (Source: {location})\n{doc.page_content[:500]}"
            )

        human_prompt = (
            f"Question: {question}\n\n"
            f"Please evaluate these {len(retrieved)} document chunks:\n\n"
            + "\n\n---\n\n".join(chunk_descriptions)
            + "\n\nGrade each chunk from index 0 to "
            + str(len(retrieved) - 1)
            + "."
        )

        batch_grader = llm.with_structured_output(BatchGradingResult)
        result: BatchGradingResult = batch_grader.invoke([
            SystemMessage(content=_BATCH_GRADE_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])

        relevant_indices = {
            item.chunk_index
            for item in result.evaluations
            if item.score == "relevant"
        }

        relevant_docs = [doc for idx, doc in enumerate(retrieved) if idx in relevant_indices]

        logger.info(
            "Batch grading complete: %d/%d chunks are relevant.",
            len(relevant_docs), len(retrieved),
        )
        return {"relevant_documents": relevant_docs}

    except Exception as e:
        logger.warning("Batch grading failed: %s — gracefully keeping retrieved chunks", e)
        # Graceful fallback: preserve retrieved chunks so generation can still answer
        return {
            "relevant_documents": retrieved,
        }
