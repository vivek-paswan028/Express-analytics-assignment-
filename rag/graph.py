"""
rag/graph.py
─────────────
LangGraph StateGraph construction and compilation.

This module wires together all four nodes plus the fallback node
using explicit edges and conditional edges. The compiled graph is
the single entry point for the entire RAG pipeline.

Graph topology:
    START → query_analysis → retrieval → grading
                 ↑                           │
                 │           [conditional]   │
                 └───── query_analysis ──────┤
                               (retry)       │
                         generate ◄──────────┤ (relevant found)
                         fallback ◄──────────┘ (retries exhausted / error)
    generate → END
    fallback → END

Usage:
    from rag.graph import get_compiled_graph
    graph = get_compiled_graph()
    result = graph.invoke({...initial state...})
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from rag.nodes.fallback import fallback_node
from rag.nodes.generation import generation_node
from rag.nodes.grading import grading_node
from rag.nodes.query_analysis import query_analysis_node
from rag.nodes.retrieval import retrieval_node
from rag.routing import route_after_grading, route_after_query_analysis
from rag.state import RAGState

logger = logging.getLogger(__name__)


def build_graph() -> object:
    """
    Construct and compile the self-corrective RAG StateGraph.

    Node registration order does not determine execution order —
    edges do. The conditional edge after grading is the core of
    the self-corrective mechanism.
    """
    builder = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("grading", grading_node)
    builder.add_node("generate", generation_node)
    builder.add_node("fallback", fallback_node)

    # ── Static edges & conditional fast path ──────────────────────────────
    builder.add_edge(START, "query_analysis")

    # Conditional edge after query analysis: conversational queries skip retrieval
    builder.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "retrieval": "retrieval",
            "fallback": "fallback",
        },
    )
    builder.add_edge("retrieval", "grading")

    # ── Conditional edge after grading ─────────────────────────────────────
    # route_after_grading() returns one of: "generate", "query_analysis", "fallback"
    builder.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generate": "generate",
            "query_analysis": "query_analysis",  # forms the retry cycle
            "fallback": "fallback",
        },
    )

    # ── Terminal edges ─────────────────────────────────────────────────────
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    graph = builder.compile()
    logger.info("RAG graph compiled successfully.")
    return graph


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Return the singleton compiled graph.
    lru_cache ensures the graph is built only once per process
    even if called from multiple request handlers.
    """
    return build_graph()


def run_rag_pipeline(
    question: str,
    top_k: int = 5,
    max_retries: int = 2,
) -> RAGState:
    """
    Convenience function: run the full RAG pipeline for a question.

    Args:
        question:    The user's question.
        top_k:       Number of chunks to retrieve per attempt.
        max_retries: Max query-rewrite + re-retrieval cycles.

    Returns:
        The final RAGState after the graph reaches END.
    """
    graph = get_compiled_graph()

    initial_state: RAGState = {
        "original_question": question,
        "top_k": top_k,
        "rewritten_query": "",
        "query_type": "unknown",
        "retrieved_documents": [],
        "relevant_documents": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "answer": "",
        "sources": [],
        "generation_supported": False,
        "error": None,
    }

    logger.info("Running RAG pipeline for: '%s'", question[:80])
    result = graph.invoke(initial_state)
    return result

