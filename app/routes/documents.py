"""
app/routes/documents.py
────────────────────────
GET /documents — list all indexed documents.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.documents import DocumentInfo, DocumentListResponse
from ingestion.vector_store import get_vector_store

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents() -> DocumentListResponse:
    """
    Return a list of all documents currently indexed in the vector store,
    grouped by document identifier with chunk counts.
    """
    store = get_vector_store()
    doc_list = store.list_documents()
    total_chunks = store.get_total_chunks()

    return DocumentListResponse(
        total_documents=len(doc_list),
        total_chunks=total_chunks,
        documents=[DocumentInfo(**d) for d in doc_list],
    )

