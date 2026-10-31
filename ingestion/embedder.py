"""
ingestion/embedder.py
──────────────────────
Local embedding model using sentence-transformers (all-MiniLM-L6-v2).

Why local embeddings instead of OpenAI/Google embeddings?
- Zero API cost — no second API key required.
- Works fully offline after the first model download.
- all-MiniLM-L6-v2 produces 384-dimensional vectors with strong
  semantic understanding for English text, sufficient for a 5-document
  technical corpus.
- Model is cached by HuggingFace in ~/.cache/huggingface after first use.

Trade-off: quality is slightly lower than OpenAI text-embedding-3-small,
but the gap is imperceptible at the scale of 5 documents and ~100 chunks.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_community.embeddings import HuggingFaceEmbeddings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    The lru_cache ensures the model is loaded only once per process,
    even if get_embedding_model() is called from multiple modules.
    First call triggers a model download if not cached locally (~90 MB).
    """
    logger.info("Loading embedding model: %s (may download on first run)", model_name)
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity friendly
    )
    logger.info("Embedding model loaded: %s", model_name)
    return model

