from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    doc_id: str = Field(description="Stable document identifier")
    source: str = Field(description="Source filename or URL")
    chunk_id: str = Field(description="Unique chunk identifier within the document")
    heading: str = Field(default="", description="Nearest section heading above this chunk")


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural language question to answer",
        examples=["How do I add a conditional edge in LangGraph?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (default: 5)",
    )

    model_config = {"json_schema_extra": {"example": {"question": "What is a StateGraph?", "top_k": 5}}}


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer, grounded in the retrieved context")
    sources: list[SourceMetadata] = Field(description="Source chunks used to generate the answer")
    query_type: str = Field(description="Classified question type")
    rewritten_query: str = Field(description="Expanded query used for retrieval")
    retry_count: int = Field(description="Number of query-rewrite cycles performed")
    relevant_chunks_used: int = Field(description="Number of relevant chunks passed to generation")
    generation_supported: bool = Field(
        description="True if the answer is grounded in indexed documentation; False if fallback was used"
    )
    error: Optional[str] = Field(default=None, description="Error message if something went wrong")

