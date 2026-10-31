"""
rag/llm.py
──────────
Factory for Google Gemini language models using the latest Google GenAI specifications:
- Uses `gemini-2.0-flash` by default.
- Reads either `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- Passes `api_key` explicitly to comply with the latest langchain-google-genai / google-genai standards.
- Sets explicit request timeouts and retry counts for resilience.
"""
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
    """
    Construct a ChatGoogleGenerativeAI client adhering to the latest Gemini API specs.

    Args:
        temperature: Sampling temperature (defaults to settings.llm_temperature).
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum automatic retries on rate limits or transient errors.
    """
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
