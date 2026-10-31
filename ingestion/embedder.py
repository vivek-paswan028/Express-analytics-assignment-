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
    logger.info("Loading embedding model: %s", model_name)
    try:
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    logger.info("Embedding model loaded: %s", model_name)
    return model

