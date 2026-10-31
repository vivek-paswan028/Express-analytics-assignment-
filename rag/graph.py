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
    builder = StateGraph(RAGState)

    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("grading", grading_node)
    builder.add_node("generate", generation_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "query_analysis")

    builder.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "retrieval": "retrieval",
            "fallback": "fallback",
        },
    )
    builder.add_edge("retrieval", "grading")

    builder.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generate": "generate",
            "query_analysis": "query_analysis",
            "fallback": "fallback",
        },
    )

    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    graph = builder.compile()
    logger.info("RAG graph compiled successfully.")
    return graph


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()


def run_rag_pipeline(
    question: str,
    top_k: int = 5,
    max_retries: int = 2,
) -> RAGState:
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

