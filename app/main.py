"""
app/main.py
────────────
FastAPI application entry point.

Startup sequence (via lifespan):
1. Validate the LLM API key is set (warn if missing, don't crash).
2. Initialise the embedding model (downloaded from HuggingFace on first run).
3. Initialise the ChromaDB vector store.
4. Pre-ingest the corpus documents if the collection is empty.
5. Compile the LangGraph workflow.

All heavy initialisation is done once at startup, not per request.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import documents, feedback, ingest, query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

CORPUS_DIR = Path(__file__).parent.parent / "docs" / "corpus"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RAG Documentation Assistant — Starting up")
    logger.info("=" * 60)

    # Validate API key
    if not settings.has_llm_key:
        logger.warning(
            "GOOGLE_API_KEY is not set. Query endpoints will return 503 until the key is configured."
        )
    else:
        logger.info("LLM: %s", settings.llm_model)

    # Initialise vector store (loads embedding model, connects to ChromaDB)
    from ingestion.vector_store import init_vector_store
    store = init_vector_store(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_model_name=settings.embedding_model,
    )

    # Auto-ingest corpus only if explicitly configured (defaults to False for clean slate)
    if settings.auto_ingest_corpus and store.get_total_chunks() == 0:
        logger.info("Auto-ingest enabled: ingesting corpus from %s", CORPUS_DIR)
        _ingest_corpus(store)
    else:
        logger.info(
            "Vector store ready: %d chunks across %d documents",
            store.get_total_chunks(),
            len(store.list_documents()),
        )

    # Compile LangGraph workflow (validates graph structure)
    from rag.graph import get_compiled_graph
    get_compiled_graph()
    logger.info("LangGraph workflow compiled and ready.")
    logger.info("=" * 60)
    logger.info("Application ready at http://%s:%d", settings.app_host, settings.app_port)
    logger.info("API docs:  http://%s:%d/docs", settings.app_host, settings.app_port)
    logger.info("=" * 60)

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("RAG Documentation Assistant — Shutting down")


def _ingest_corpus(store) -> None:
    """Ingest all documents from the corpus directory at startup."""
    from ingestion.chunker import chunk_documents
    from ingestion.loader import load_corpus

    if not CORPUS_DIR.exists():
        logger.warning("Corpus directory not found: %s", CORPUS_DIR)
        return

    try:
        docs = load_corpus(CORPUS_DIR)
        if not docs:
            logger.warning("No documents found in corpus directory.")
            return

        chunks = chunk_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        result = store.add_documents(chunks)
        logger.info(
            "Corpus ingested: %d chunks added, %d skipped (duplicates)",
            result["added"], result["skipped_duplicate"],
        )
    except Exception as e:
        logger.error("Corpus ingestion failed: %s", e)


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Technical Documentation Assistant",
        description=(
            "A self-corrective RAG system built with LangGraph and FastAPI. "
            "Answers questions about indexed technical documentation with citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for local development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(query.router, tags=["Query"])
    app.include_router(ingest.router, tags=["Ingestion"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(feedback.router, tags=["Feedback"])

    @app.get("/health", tags=["Health"])
    def health_check():
        """Quick health check — confirms the app is running."""
        from ingestion.vector_store import get_vector_store
        try:
            store = get_vector_store()
            return {
                "status": "healthy",
                "llm_configured": settings.has_llm_key,
                "llm_model": settings.llm_model,
                "indexed_chunks": store.get_total_chunks(),
                "indexed_documents": len(store.list_documents()),
            }
        except Exception as e:
            return {"status": "degraded", "error": str(e)}

    return app


app = create_app()

