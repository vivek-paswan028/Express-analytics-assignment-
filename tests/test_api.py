"""
tests/test_api.py
──────────────────
FastAPI endpoint tests using TestClient.
LLM calls and vector store are mocked so these tests run without API keys.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


def _make_mock_store(chunks=5, docs=None):
    """Build a mock VectorStore."""
    mock = MagicMock()
    mock.get_total_chunks.return_value = chunks
    mock.list_documents.return_value = docs or [
        {"doc_id": "doc_001", "source": "langgraph_concepts.md",
         "title": "LangGraph", "chunks": chunks, "ingested_at": "2025-01-01T00:00:00Z"}
    ]
    mock.similarity_search.return_value = [
        Document(
            page_content="LangGraph StateGraph has nodes and conditional edges.",
            metadata={"source": "langgraph_concepts.md", "doc_id": "doc_001",
                      "chunk_id": "doc_001_chunk_0001", "heading": "Nodes"},
        )
    ]
    mock.add_documents.return_value = {"added": 3, "skipped_duplicate": 0}
    return mock


def _make_mock_rag_result(
    answer="LangGraph uses StateGraph with nodes.",
    sources=None,
    retry_count=0,
    generation_supported=True,
):
    """Build a mock RAG pipeline result dict."""
    default_sources = [
        {"doc_id": "doc_001", "source": "langgraph_concepts.md",
         "chunk_id": "doc_001_chunk_0001", "heading": "Nodes"}
    ]
    return {
        "original_question": "What is LangGraph?",
        "rewritten_query": "LangGraph StateGraph nodes edges",
        "query_type": "conceptual",
        "retrieved_documents": [MagicMock()],
        "relevant_documents": [MagicMock()],
        "retry_count": retry_count,
        "max_retries": 2,
        "answer": answer,
        "sources": sources if sources is not None else default_sources,
        "generation_supported": generation_supported,
        "error": None,
    }


@pytest.fixture
def client():
    """Create a TestClient with all dependencies mocked."""
    mock_store = _make_mock_store()

    with patch("ingestion.vector_store.init_vector_store", return_value=mock_store), \
         patch("ingestion.vector_store.get_vector_store", return_value=mock_store), \
         patch("ingestion.embedder.get_embedding_model"), \
         patch("rag.graph.get_compiled_graph"), \
         patch("app.config.get_settings") as mock_settings_fn:

        mock_settings = MagicMock()
        mock_settings.has_llm_key = True
        mock_settings.llm_model = "gemini-2.0-flash"
        mock_settings.llm_temperature = 0.0
        mock_settings.google_api_key = "test_key"
        mock_settings.chroma_persist_dir = "/tmp/test_chroma"
        mock_settings.chroma_collection_name = "test"
        mock_settings.embedding_model = "all-MiniLM-L6-v2"
        mock_settings.retrieval_top_k = 5
        mock_settings.max_retries = 2
        mock_settings.chunk_size = 800
        mock_settings.chunk_overlap = 100
        mock_settings.max_upload_size_mb = 5
        mock_settings.app_host = "0.0.0.0"
        mock_settings.app_port = 8000
        mock_settings.app_debug = False
        tmp_feedback = Path(tempfile.mktemp(suffix=".jsonl"))
        mock_settings.feedback_log_file = tmp_feedback
        mock_settings_fn.return_value = mock_settings

        from app.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── POST /query ───────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_valid_question_returns_200(self, client):
        mock_result = _make_mock_rag_result()
        with patch("app.routes.query.run_rag_pipeline", return_value=mock_result), \
             patch("app.routes.query.get_settings") as ms:
            ms.return_value.has_llm_key = True
            ms.return_value.max_retries = 2
            response = client.post("/query", json={"question": "What is LangGraph?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "retry_count" in data
        assert "generation_supported" in data

    def test_empty_question_returns_422(self, client):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_returns_422(self, client):
        response = client.post("/query", json={"top_k": 5})
        assert response.status_code == 422

    def test_top_k_out_of_range_returns_422(self, client):
        response = client.post("/query", json={"question": "test", "top_k": 100})
        assert response.status_code == 422

    def test_fallback_response_is_valid(self, client):
        """Fallback (no relevant docs) still returns 200 with generation_supported=False."""
        mock_result = _make_mock_rag_result(
            answer="I could not find sufficient information...",
            sources=[],
            retry_count=2,
            generation_supported=False,
        )
        # Override relevant_documents so the route reports 0 relevant chunks
        mock_result["relevant_documents"] = []
        with patch("app.routes.query.run_rag_pipeline", return_value=mock_result), \
             patch("app.routes.query.get_settings") as ms:
            ms.return_value.has_llm_key = True
            ms.return_value.max_retries = 2
            response = client.post("/query", json={"question": "Tell me about quantum entanglement"})
        assert response.status_code == 200
        data = response.json()
        assert data["generation_supported"] is False
        assert data["sources"] == []

# ── GET /documents ────────────────────────────────────────────────────────────

class TestDocumentsEndpoint:
    def test_returns_document_list(self, client):
        mock_store = _make_mock_store()
        with patch("app.routes.documents.get_vector_store", return_value=mock_store):
            response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_empty_store_returns_empty_list(self, client):
        empty_store = MagicMock()
        empty_store.get_total_chunks.return_value = 0
        empty_store.list_documents.return_value = []
        with patch("app.routes.documents.get_vector_store", return_value=empty_store):
            response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0
        assert data["documents"] == []

# ── POST /feedback ────────────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    def test_thumbs_up_accepted(self, client):
        with patch("app.routes.feedback.get_settings") as ms:
            tmp = Path(tempfile.mktemp(suffix=".jsonl"))
            ms.return_value.feedback_log_file = tmp
            response = client.post("/feedback", json={
                "rating": "thumbs_up",
                "comment": "Great answer!",
                "question": "What is ChromaDB?",
            })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert "feedback_id" in data
        assert data["feedback_id"].startswith("fb_")

    def test_thumbs_down_accepted(self, client):
        with patch("app.routes.feedback.get_settings") as ms:
            tmp = Path(tempfile.mktemp(suffix=".jsonl"))
            ms.return_value.feedback_log_file = tmp
            response = client.post("/feedback", json={"rating": "thumbs_down"})
        assert response.status_code == 200

    def test_invalid_rating_returns_422(self, client):
        response = client.post("/feedback", json={"rating": "neutral"})
        assert response.status_code == 422

    def test_missing_rating_returns_422(self, client):
        response = client.post("/feedback", json={"comment": "nice"})
        assert response.status_code == 422

    def test_feedback_written_to_file(self, client):
        """Feedback is appended to the JSONL log file."""
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            tmp_path = Path(f.name)

        try:
            # Patch the module-level settings object used inside the route
            import app.routes.feedback as feedback_module
            original_settings = feedback_module.settings
            mock_s = MagicMock()
            mock_s.feedback_log_file = tmp_path
            feedback_module.settings = mock_s

            response = client.post("/feedback", json={
                "rating": "thumbs_up",
                "question": "Test question",
                "comment": "Test comment",
            })
            assert response.status_code == 200
        finally:
            feedback_module.settings = original_settings

        content = tmp_path.read_text().strip()
        assert content, "Feedback file should not be empty"
        record = json.loads(content)
        assert record["rating"] == "thumbs_up"
        assert record["comment"] == "Test comment"
        assert "feedback_id" in record
        assert "timestamp" in record
        tmp_path.unlink(missing_ok=True)


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        mock_store = _make_mock_store()
        with patch("app.main.get_vector_store", return_value=mock_store, create=True):
            response = client.get("/health")
        # Health endpoint should always return 200
        assert response.status_code == 200


# ── POST /ingest ──────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_ingest_markdown_file(self, client):
        """Upload a valid markdown file and verify it's ingested."""
        md_content = b"# Test Doc\n\nThis is test content for ingestion."
        mock_store = _make_mock_store()

        with patch("app.routes.ingest.get_vector_store", return_value=mock_store), \
             patch("app.routes.ingest.get_settings") as ms:
            ms.return_value.max_upload_size_mb = 5
            ms.return_value.chunk_size = 800
            ms.return_value.chunk_overlap = 100

            response = client.post(
                "/ingest",
                files=[("files", ("test.md", md_content, "text/markdown"))],
            )

        assert response.status_code == 200
        data = response.json()
        assert "ingested" in data
        assert "errors" in data

    def test_unsupported_file_type_returns_error(self, client):
        """CSV files are not supported; error is in the errors list."""
        csv_content = b"a,b,c\n1,2,3"
        response = client.post(
            "/ingest",
            files=[("files", ("data.csv", csv_content, "text/csv"))],
        )
        assert response.status_code == 200  # HTTP 200 with error in body
        data = response.json()
        assert len(data["errors"]) > 0
        assert "Unsupported" in data["errors"][0]["error"]

    def test_invalid_url_returns_422(self, client):
        """FTP URLs are rejected at validation."""
        response = client.post("/ingest/url", json={"urls": ["ftp://example.com/file.txt"]})
        assert response.status_code == 422
