from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.ingest import IngestError, IngestResponse, IngestResult, URLIngestRequest
from ingestion.chunker import chunk_documents
from ingestion.loader import _make_doc_id, load_file, load_url
from ingestion.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

_ALLOWED_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm", ".pdf"}


@router.post("/ingest", response_model=IngestResponse, summary="Ingest documents by file upload")
async def ingest_files(
    files: list[UploadFile] = File(..., description="One or more document files to ingest"),
) -> IngestResponse:
    store = get_vector_store()
    results: list[IngestResult] = []
    errors: list[IngestError] = []
    total_added = 0

    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    for upload in files:
        filename = upload.filename or "unknown"
        ext = Path(filename).suffix.lower()

        if ext not in _ALLOWED_EXTENSIONS:
            errors.append(IngestError(
                source=filename,
                error=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            ))
            continue

        try:
            content = await upload.read()

            if len(content) > max_bytes:
                errors.append(IngestError(
                    source=filename,
                    error=f"File exceeds {settings.max_upload_size_mb} MB limit.",
                ))
                continue

            if not content.strip():
                errors.append(IngestError(source=filename, error="File is empty."))
                continue

            with tempfile.NamedTemporaryFile(
                suffix=ext, delete=False, prefix="upload_"
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                docs = load_file(tmp_path)
                real_doc_id = _make_doc_id(filename)
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["doc_id"] = real_doc_id
            finally:
                tmp_path.unlink(missing_ok=True)

            chunks = chunk_documents(
                docs,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            add_result = store.add_documents(chunks)
            doc_id = docs[0].metadata.get("doc_id", filename) if docs else filename

            results.append(IngestResult(
                source=filename,
                doc_id=doc_id,
                chunks_added=add_result["added"],
                chunks_skipped_duplicate=add_result["skipped_duplicate"],
            ))
            total_added += add_result["added"]
            logger.info("Ingested file: %s (%d chunks added)", filename, add_result["added"])

        except Exception as e:
            logger.error("Failed to ingest file %s: %s", filename, e)
            errors.append(IngestError(source=filename, error=str(e)))

    return IngestResponse(ingested=results, errors=errors, total_chunks_added=total_added)


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    summary="Ingest documents from URLs",
)
def ingest_urls(request: URLIngestRequest) -> IngestResponse:
    store = get_vector_store()
    results: list[IngestResult] = []
    errors: list[IngestError] = []
    total_added = 0

    for url in request.urls:
        try:
            docs = load_url(url)
            chunks = chunk_documents(
                docs,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            add_result = store.add_documents(chunks)
            doc_id = docs[0].metadata.get("doc_id", url) if docs else url

            results.append(IngestResult(
                source=url,
                doc_id=doc_id,
                chunks_added=add_result["added"],
                chunks_skipped_duplicate=add_result["skipped_duplicate"],
            ))
            total_added += add_result["added"]

        except Exception as e:
            logger.error("Failed to ingest URL %s: %s", url, e)
            errors.append(IngestError(source=url, error=str(e)))

    return IngestResponse(ingested=results, errors=errors, total_chunks_added=total_added)

