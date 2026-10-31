"""
app/routes/feedback.py
───────────────────────
POST /feedback — record thumbs-up/down feedback on an answer.

Persistence strategy: feedback is appended to a JSONL file (one JSON
object per line). This is appropriate for a demo/intern project — it's
simple, crash-safe (append-only), and human-readable.

What feedback is NOT used for:
- Automatic model retraining (this system uses API-hosted LLMs).
- Modifying retrieval weights or rankings in real time.

What feedback COULD be used for (future improvement):
- Offline analysis to identify poorly answered question categories.
- Fine-tuning a local grading or generation model.
- Flagging answers for human review.
"""
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
    """
    Record a thumbs-up or thumbs-down rating for an answer, with an
    optional free-text comment.

    Feedback is stored in an append-only JSONL log file.
    """
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

