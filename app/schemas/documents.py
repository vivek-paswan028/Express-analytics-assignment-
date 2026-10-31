"""
app/schemas/documents.py
─────────────────────────
Response schema for GET /documents.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    title: str = ""
    chunks: int
    ingested_at: str = ""


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]

