from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    rating: str = Field(
        ...,
        description="'thumbs_up' or 'thumbs_down'",
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text comment",
    )
    query_id: Optional[str] = Field(
        default=None,
        description="Optional identifier linking feedback to a specific query",
    )
    question: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="The question this feedback is about (if query_id is not available)",
    )

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: str) -> str:
        allowed = {"thumbs_up", "thumbs_down"}
        if v not in allowed:
            raise ValueError(f"rating must be 'thumbs_up' or 'thumbs_down', got '{v}'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "rating": "thumbs_up",
                "comment": "The answer was accurate and cited the correct section.",
                "question": "How do I add a node in LangGraph?",
            }
        }
    }


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    feedback_id: str
    message: str = "Thank you for your feedback."

