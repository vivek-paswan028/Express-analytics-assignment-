"""
rag/state.py
─────────────
LangGraph state schema for the self-corrective RAG workflow.

Design rationale:
- TypedDict is used (not Pydantic BaseModel) because LangGraph's StateGraph
  works natively with TypedDict and handles reducer functions via Annotated.
- The schema documents every field's purpose and lifecycle.
- retry_count and max_retries are both in state so the routing function
  is a pure function of state (no global config needed at routing time).
"""
from __future__ import annotations

from typing import Any, Optional
from typing import TypedDict

from langchain_core.documents import Document


class SourceMetadata(TypedDict):
    """Citation information for one chunk used in the answer."""
    doc_id: str
    source: str       # filename or URL
    chunk_id: str
    heading: str      # nearest section heading in the source document


class RAGState(TypedDict):
    """
    Complete state that flows through every node of the RAG workflow.

    Fields are grouped by lifecycle phase:
    ┌────────────────────────────────────────────────────────────┐
    │ INPUT          Set at entry; never mutated.               │
    │ QUERY          Updated by query_analysis (and rewrite).   │
    │ RETRIEVAL      Updated by retrieval node.                 │
    │ GRADING        Updated by grading node.                   │
    │ SELF-CORRECT   Updated by routing / rewrite logic.        │
    │ OUTPUT         Set by generation or fallback node.        │
    │ ERROR          Set if any node encounters an exception.   │
    └────────────────────────────────────────────────────────────┘
    """

    # ── INPUT ─────────────────────────────────────────────────────────────
    original_question: str
    """The raw question exactly as submitted by the user. Never modified."""

    top_k: int
    """Number of chunks to retrieve. Passed from the API request."""

    # ── QUERY PROCESSING ──────────────────────────────────────────────────
    rewritten_query: str
    """
    The query sent to the vector store. On the first pass this is the
    rewritten/expanded version of original_question. On subsequent retries
    it is a further reformulation produced by the query_analysis node.
    """

    query_type: str
    """
    Classification of the question: 'conceptual' | 'how-to' |
    'troubleshooting' | 'api-reference' | 'unknown'.
    Used as a hint for retrieval expansion and prompt construction.
    """

    # ── RETRIEVAL ─────────────────────────────────────────────────────────
    retrieved_documents: list[Document]
    """Raw chunks returned by the vector store for the current query."""

    # ── GRADING ───────────────────────────────────────────────────────────
    relevant_documents: list[Document]
    """
    Subset of retrieved_documents graded 'relevant' by the LLM.
    If empty after grading, routing triggers a retry or fallback.
    """

    # ── SELF-CORRECTION ───────────────────────────────────────────────────
    retry_count: int
    """
    Number of query-rewrite + re-retrieval cycles completed so far.
    Starts at 0. Incremented by the query_analysis node when it detects
    it is running in retry mode (i.e., relevant_documents is empty).
    """

    max_retries: int
    """
    Hard ceiling on retry_count. When retry_count >= max_retries the
    routing function sends execution to the fallback node instead of
    looping back. Default: 2 (see config.py for rationale).
    """

    # ── OUTPUT ────────────────────────────────────────────────────────────
    answer: str
    """
    The final answer returned to the user.
    Set by the generation node (grounded answer) or the fallback node
    ("I don't know" message).
    """

    sources: list[SourceMetadata]
    """
    Citation list included in the response. Each entry references one
    chunk from relevant_documents that contributed to the answer.
    Empty when the fallback path is taken.
    """

    generation_supported: bool
    """
    True  → answer is grounded in relevant retrieved context.
    False → fallback path was taken (no sufficient evidence found).
    """

    # ── ERROR ─────────────────────────────────────────────────────────────
    error: Optional[str]
    """
    If a node catches an unrecoverable exception it stores the message
    here and the graph routes to the fallback node. API callers receive
    a 503 with the user-facing portion of this message.
    Never expose raw stack traces through this field.
    """

