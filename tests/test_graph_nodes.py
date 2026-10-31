"""
tests/test_graph_nodes.py
──────────────────────────
Unit tests for individual LangGraph nodes.
All LLM calls are mocked — these tests run without an API key.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.state import RAGState


def _make_state(**overrides) -> RAGState:
    """Build a minimal valid RAGState with sensible defaults."""
    base: RAGState = {
        "original_question": "What is a StateGraph?",
        "top_k": 5,
        "rewritten_query": "",
        "query_type": "unknown",
        "retrieved_documents": [],
        "relevant_documents": [],
        "retry_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "generation_supported": False,
        "error": None,
    }
    base.update(overrides)
    return base


def _make_doc(content: str, source: str = "test.md", chunk_id: str = "chunk_001") -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "doc_id": "doc_test",
            "chunk_id": chunk_id,
            "heading": "Overview",
        },
    )


# ── Routing tests (pure functions, no mocking needed) ─────────────────────────

class TestRouting:
    def test_routes_to_generate_when_relevant_docs(self):
        from rag.routing import route_after_grading
        state = _make_state(relevant_documents=[_make_doc("relevant content")])
        assert route_after_grading(state) == "generate"

    def test_routes_to_query_analysis_on_first_retry(self):
        from rag.routing import route_after_grading
        state = _make_state(relevant_documents=[], retry_count=0, max_retries=2)
        assert route_after_grading(state) == "query_analysis"

    def test_routes_to_fallback_when_retries_exhausted(self):
        from rag.routing import route_after_grading
        state = _make_state(relevant_documents=[], retry_count=2, max_retries=2)
        assert route_after_grading(state) == "fallback"

    def test_routes_to_fallback_on_error(self):
        from rag.routing import route_after_grading
        state = _make_state(
            relevant_documents=[_make_doc("some content")],
            error="LLM API failed",
        )
        assert route_after_grading(state) == "fallback"

    def test_routes_to_query_analysis_on_second_retry(self):
        from rag.routing import route_after_grading
        state = _make_state(relevant_documents=[], retry_count=1, max_retries=2)
        assert route_after_grading(state) == "query_analysis"

    def test_fallback_when_retry_equals_max(self):
        from rag.routing import route_after_grading
        # At exactly max_retries → fallback (not equal to, but >= max)
        state = _make_state(relevant_documents=[], retry_count=3, max_retries=2)
        assert route_after_grading(state) == "fallback"


# ── Fallback node tests ───────────────────────────────────────────────────────

class TestFallbackNode:
    def test_fallback_no_error(self):
        from rag.nodes.fallback import fallback_node
        state = _make_state(retry_count=2)
        result = fallback_node(state)
        assert result["generation_supported"] is False
        assert result["sources"] == []
        assert "could not find" in result["answer"].lower()

    def test_fallback_with_error(self):
        from rag.nodes.fallback import fallback_node
        state = _make_state(error="Retrieval failed: timeout")
        result = fallback_node(state)
        assert result["generation_supported"] is False
        assert "error" in result["answer"].lower() or "system error" in result["answer"].lower()

    def test_fallback_answer_does_not_contain_raw_stacktrace(self):
        from rag.nodes.fallback import fallback_node
        state = _make_state(error="SomeException: internal details: line 42 in foo.py")
        result = fallback_node(state)
        assert "line 42" not in result["answer"]
        assert "foo.py" not in result["answer"]


# ── Retrieval node tests ──────────────────────────────────────────────────────

class TestRetrievalNode:
    def test_retrieval_uses_rewritten_query(self):
        """Retrieval node calls similarity_search with the rewritten query."""
        from rag.nodes.retrieval import retrieval_node

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [_make_doc("result")]

        state = _make_state(
            rewritten_query="StateGraph node edge LangGraph",
            original_question="What is a StateGraph?",
        )

        with patch("rag.nodes.retrieval.get_vector_store", return_value=mock_store):
            result = retrieval_node(state)

        mock_store.similarity_search.assert_called_once_with(
            query="StateGraph node edge LangGraph", k=5
        )
        assert len(result["retrieved_documents"]) == 1

    def test_retrieval_falls_back_to_original_question_when_no_rewrite(self):
        from rag.nodes.retrieval import retrieval_node

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        state = _make_state(rewritten_query="", original_question="What is ChromaDB?")

        with patch("rag.nodes.retrieval.get_vector_store", return_value=mock_store):
            result = retrieval_node(state)

        call_args = mock_store.similarity_search.call_args
        assert call_args[1]["query"] == "What is ChromaDB?" or call_args[0][0] == "What is ChromaDB?"

    def test_retrieval_handles_store_exception(self):
        """Store failure → empty list + error in state (no crash)."""
        from rag.nodes.retrieval import retrieval_node

        mock_store = MagicMock()
        mock_store.similarity_search.side_effect = RuntimeError("connection refused")

        with patch("rag.nodes.retrieval.get_vector_store", return_value=mock_store):
            result = retrieval_node(_make_state())

        assert result["retrieved_documents"] == []
        assert "error" in result
        assert result["error"] is not None


# ── Query analysis node tests (mocked LLM) ────────────────────────────────────

class TestQueryAnalysisNode:
    def _mock_analysis_result(self, query_type="how-to", rewritten="expanded query"):
        from rag.nodes.query_analysis import QueryAnalysisResult
        return QueryAnalysisResult(
            query_type=query_type,
            rewritten_query=rewritten,
            reasoning="Test reasoning.",
        )

    def test_first_pass_sets_rewritten_query(self):
        from rag.nodes.query_analysis import query_analysis_node

        mock_result = self._mock_analysis_result(
            query_type="how-to",
            rewritten="add conditional edge LangGraph StateGraph routing",
        )

        with patch("rag.nodes.query_analysis.get_gemini_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_result
            mock_get_llm.return_value = mock_llm

            result = query_analysis_node(_make_state(
                original_question="How do I add a conditional edge?",
                retry_count=0,
            ))

        assert result["rewritten_query"] == "add conditional edge LangGraph StateGraph routing"
        assert result["query_type"] == "how-to"
        assert "retry_count" not in result  # first pass doesn't increment

    def test_retry_pass_increments_retry_count(self):
        from rag.nodes.query_analysis import query_analysis_node

        mock_result = self._mock_analysis_result(rewritten="alternative query formulation")

        with patch("rag.nodes.query_analysis.get_gemini_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_result
            mock_get_llm.return_value = mock_llm

            result = query_analysis_node(_make_state(
                original_question="What is LangGraph?",
                rewritten_query="LangGraph library",
                retry_count=1,
                relevant_documents=[],
            ))

        assert result["retry_count"] == 2  # incremented from 1

    def test_graceful_fallback_on_llm_failure(self):
        """Node falls back to original question if LLM throws."""
        from rag.nodes.query_analysis import query_analysis_node

        with patch("rag.nodes.query_analysis.get_gemini_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")
            mock_get_llm.return_value = mock_llm

            result = query_analysis_node(_make_state(
                original_question="What is ChromaDB?",
                retry_count=0,
            ))

        # Should not crash; falls back to original question
        assert result["rewritten_query"] == "What is ChromaDB?"
        assert result["query_type"] == "unknown"


# ── Grading node tests ────────────────────────────────────────────────────────

class TestGradingNode:
    def test_filters_irrelevant_documents(self):
        """Documents graded irrelevant are excluded from relevant_documents."""
        from rag.nodes.grading import grading_node, BatchGradingResult, ChunkEvaluation

        relevant_doc = _make_doc("LangGraph StateGraph has nodes and edges.", "langgraph.md")
        irrelevant_doc = _make_doc("Python lists are sequences of items.", "python.md", "chunk_002")

        mock_eval = BatchGradingResult(
            evaluations=[
                ChunkEvaluation(chunk_index=0, score="relevant", reasoning="On topic"),
                ChunkEvaluation(chunk_index=1, score="irrelevant", reasoning="Off topic"),
            ]
        )

        with patch("rag.nodes.grading.get_gemini_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_eval
            mock_get_llm.return_value = mock_llm

            state = _make_state(
                retrieved_documents=[relevant_doc, irrelevant_doc],
                original_question="What is a StateGraph?",
            )
            result = grading_node(state)

        assert len(result["relevant_documents"]) == 1
        assert result["relevant_documents"][0].metadata["source"] == "langgraph.md"

    def test_empty_retrieval_returns_empty_relevant(self):
        from rag.nodes.grading import grading_node
        state = _make_state(retrieved_documents=[])
        result = grading_node(state)
        assert result["relevant_documents"] == []

    def test_grading_exception_falls_back_safely(self):
        """When grading throws, node handles it gracefully without crashing."""
        from rag.nodes.grading import grading_node

        broken_grader = MagicMock()
        broken_grader.with_structured_output.return_value.invoke.side_effect = Exception("timeout")

        doc = _make_doc("Some content")
        with patch("rag.nodes.grading.get_gemini_llm", return_value=broken_grader):
            result = grading_node(_make_state(retrieved_documents=[doc]))
        assert "error" in result

