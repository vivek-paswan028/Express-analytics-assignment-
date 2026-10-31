from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class URLIngestRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="List of URLs to fetch and ingest",
    )
    doc_id_prefix: Optional[str] = Field(
        default=None,
        description="Optional prefix for doc_id (e.g. 'custom'). If omitted, derived from filename.",
    )

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, urls: list[str]) -> list[str]:
        for url in urls:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL (must start with http:// or https://): {url}")
        if len(urls) > 10:
            raise ValueError("Maximum 10 URLs per request.")
        return urls

    model_config = {
        "json_schema_extra": {
            "example": {
                "urls": ["https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md"]
            }
        }
    }


class IngestResult(BaseModel):
    source: str
    doc_id: str
    chunks_added: int
    chunks_skipped_duplicate: int


class IngestError(BaseModel):
    source: str
    error: str


class IngestResponse(BaseModel):
    ingested: list[IngestResult] = Field(default_factory=list)
    errors: list[IngestError] = Field(default_factory=list)
    total_chunks_added: int = 0

