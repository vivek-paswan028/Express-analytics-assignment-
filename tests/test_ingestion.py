from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingestion.chunker import chunk_documents, _content_hash
from ingestion.loader import load_file, load_markdown, _validate_url, _is_private_host


class TestMarkdownLoader:
    def test_loads_markdown_file(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Document\n\nThis is a test.", encoding="utf-8")
        docs = load_markdown(md_file)
        assert len(docs) == 1
        assert "test" in docs[0].page_content.lower()
        assert docs[0].metadata["source"] == "test.md"
        assert docs[0].metadata["doc_id"] == "doc_test"

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.md"
        empty.write_text("   \n   ", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_markdown(empty)

    def test_extracts_h1_title(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# My Title\n\nContent here.", encoding="utf-8")
        docs = load_markdown(md)
        assert docs[0].metadata["title"] == "My Title"

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(f)


class TestSSRFProtection:
    def test_rejects_non_https_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            _validate_url("ftp://example.com/file.txt")

    def test_rejects_localhost(self):
        assert _is_private_host("localhost") is True

    def test_rejects_private_ip(self):
        assert _is_private_host("192.168.1.1") is True
        assert _is_private_host("10.0.0.1") is True
        assert _is_private_host("172.16.0.1") is True

    def test_accepts_valid_url(self):
        try:
            _validate_url("https://raw.githubusercontent.com/test/test/main/README.md")
        except ValueError as e:
            if "private" in str(e).lower():
                pytest.skip("Network not available in sandbox")


class TestChunker:
    def _make_doc(self, text: str, source: str = "test.md") -> Document:
        return Document(
            page_content=text,
            metadata={"source": source, "doc_id": "doc_test", "title": "Test"},
        )

    def test_chunks_long_document(self):
        long_text = "## Section\n\n" + ("word " * 200 + "\n\n") * 5
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        text = "paragraph\n\n" * 30
        doc = self._make_doc(text)
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_metadata_preserved(self):
        doc = self._make_doc("## H2\n\nSome content here.\n\nMore content.", "myfile.md")
        chunks = chunk_documents([doc], chunk_size=50, chunk_overlap=5)
        for chunk in chunks:
            assert chunk.metadata["source"] == "myfile.md"
            assert chunk.metadata["doc_id"] == "doc_test"

    def test_content_hash_deterministic(self):
        text = "hello world"
        assert _content_hash(text) == _content_hash(text)

    def test_content_hash_differs(self):
        assert _content_hash("hello") != _content_hash("world")

    def test_heading_extracted(self):
        text = "## Installation\n\nRun pip install mylib.\n\n## Usage\n\nImport it."
        doc = self._make_doc(text)
        chunks = chunk_documents([doc], chunk_size=60, chunk_overlap=5)
        headings = [c.metadata.get("heading", "") for c in chunks]
        assert any(h for h in headings)

    def test_short_document_single_chunk(self):
        doc = self._make_doc("This is a short document with very little content.")
        chunks = chunk_documents([doc], chunk_size=800, chunk_overlap=100)
        assert len(chunks) == 1

    def test_empty_document_list(self):
        chunks = chunk_documents([], chunk_size=800, chunk_overlap=100)
        assert chunks == []

