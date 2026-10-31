from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas.query import QueryRequest, QueryResponse, SourceMetadata
from rag.graph import run_rag_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
def query_documents(request: QueryRequest) -> QueryResponse:
    if not settings.has_llm_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GOOGLE_API_KEY is not configured. "
                "Set it in your .env file and restart the application."
            ),
        )

    try:
        result = run_rag_pipeline(
            question=request.question,
            top_k=request.top_k,
            max_retries=settings.max_retries,
        )

        sources = [
            SourceMetadata(
                doc_id=s.get("doc_id", ""),
                source=s.get("source", ""),
                chunk_id=s.get("chunk_id", ""),
                heading=s.get("heading", ""),
            )
            for s in (result.get("sources") or [])
        ]

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            query_type=result.get("query_type", "unknown"),
            rewritten_query=result.get("rewritten_query", request.question),
            retry_count=result.get("retry_count", 0),
            relevant_chunks_used=len(result.get("relevant_documents") or []),
            generation_supported=result.get("generation_supported", False),
            error=result.get("error"),
        )

    except Exception as e:
        logger.exception("Unhandled error in /query: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )

