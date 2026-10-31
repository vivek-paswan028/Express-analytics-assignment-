from __future__ import annotations

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings

logger = logging.getLogger(__name__)


def get_gemini_llm(
    temperature: Optional[float] = None,
    timeout: float = 45.0,
    max_retries: int = 2,
) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    api_key = settings.effective_api_key

    if not api_key:
        logger.warning("No Gemini API key found (checked GEMINI_API_KEY and GOOGLE_API_KEY).")

    temp = temperature if temperature is not None else settings.llm_temperature

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temp,
        api_key=api_key or "missing_key",
        timeout=timeout,
        max_retries=max_retries,
    )

