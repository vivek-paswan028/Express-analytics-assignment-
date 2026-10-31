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


@router.delete("/documents", summary="Delete all indexed documents")
def clear_all_documents() -> dict[str, str | int]:
    """Wipe out all indexed documents and chunks from the vector store."""
    store = get_vector_store()
    deleted_chunks = store.clear_all_documents()
    return {
        "status": "success",
        "message": f"Successfully deleted all {deleted_chunks} chunks from the vector store.",
        "deleted_chunks": deleted_chunks,
    }


@router.delete("/documents/{doc_id}", summary="Delete a document by doc_id")
def delete_document(doc_id: str) -> dict[str, str | int]:
    """Delete a specific document and all its chunks by its doc_id."""
    store = get_vector_store()
    deleted = store.delete_document(doc_id)
    return {
        "status": "success",
        "doc_id": doc_id,
        "deleted_chunks": deleted,
    }

