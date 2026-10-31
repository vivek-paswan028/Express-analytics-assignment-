from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit answer feedback")
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
    record = {
        "feedback_id": feedback_id,
        "rating": request.rating,
        "comment": request.comment,
        "query_id": request.query_id,
        "question": request.question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        log_file: Path = settings.feedback_log_file
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Feedback recorded: %s (%s)", feedback_id, request.rating)
    except Exception as e:
        logger.error("Failed to write feedback: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to record feedback. Please try again.",
        )

    return FeedbackResponse(
        status="recorded",
        feedback_id=feedback_id,
        message="Thank you for your feedback.",
    )

