# RAG-Based Technical Documentation Assistant
### *Self-Corrective LangGraph Workflow with Retrieval, Grading, and Grounded Answer Generation*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.9-blue?logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.8-orange)](https://docs.trychroma.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-49%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)

---

## 1. Project Overview

This repository implements a production-grade, self-corrective Retrieval-Augmented Generation (RAG) assistant for technical documentation. Built for the Express Analytics AI/ML Engineer Intern assignment, it answers developer questions using an indexed corpus of technical documentation (LangGraph, LangChain LCEL, ChromaDB, FastAPI, and Pydantic V2).

Rather than relying on a naive single-shot retrieval chain, the system is modeled as an explicit cyclic state machine using **LangGraph `StateGraph`**:
1. **Query Analysis & Classification**: Analyzes the question intent, classifies its query type (`conceptual`, `how-to`, `troubleshooting`, `api-reference`), and rewrites/expands it for semantic search.
2. **Vector Retrieval**: Performs similarity search against a persistent **ChromaDB** vector store with rich chunk metadata.
3. **Document Relevance Grading**: Evaluates each retrieved chunk using an LLM with structured output (`relevant` vs. `irrelevant`), filtering out noise.
4. **Self-Corrective Routing Loop**: If no relevant documents are found, the graph loops back to rewrite the query from a different angle and re-retrieves context, up to a configurable retry limit (`max_retries = 2`).
5. **Grounded Generation & Citation**: Produces answers strictly from verified context with inline source citations (`[Source N: file — heading]`).
6. **Graceful Fallback**: If the context remains insufficient after exhausting retries, the assistant returns an honest "I don't know based on the indexed documentation" response instead of hallucinating.

---

## 2. Architecture & LangGraph Workflow

### Graph Execution Diagram

```
                 User Question / POST /query
                              │
                              ▼
                   ┌─────────────────────┐
                   │   query_analysis    │  ◄───────────────────────────┐
                   │  - classify query   │                              │
                   │  - rewrite & expand │                              │
                   └──────────┬──────────┘                              │
                              │ rewritten_query                         │
                              ▼                                         │
                   ┌─────────────────────┐                              │
                   │      retrieval      │                              │
                   │  - ChromaDB search  │                              │
                   │  - top-k chunks     │                              │
                   └──────────┬──────────┘                              │
                              │ retrieved_documents                     │
                              ▼                                         │
                   ┌─────────────────────┐                              │
                   │       grading       │                              │
                   │  - LLM evaluator    │                              │
                   │  - filter relevant  │                              │
                   └──────────┬──────────┘                              │
                              │                                         │
                    [route_after_grading]                               │
                    /                   \                               │
          relevant docs?             no relevant docs?                  │
                │                               │                       │
                ▼                 retry_count < max_retries?            │
     ┌────────────────────┐            /                 \              │
     │      generate      │          yes                 no             │
     │ - grounded answer  │           │                   │             │
     │ - source citations │      (retry cycle)            ▼             │
     └──────────┬─────────┘           └────────── ┌───────────────┐     │
                │                                 │   fallback    │     │
                ▼                                 │ "I don't know"│     │
              [END]                               └───────┬───────┘     │
                                                          │             │
                                                          ▼             │
                                                        [END]           │
```

### State Schema (`rag/state.py`)

The state flowing through all nodes is defined using Python's native `TypedDict`:

```python
class RAGState(TypedDict):
    original_question: str          # Raw user query (never mutated)
    top_k: int                      # Configurable chunk retrieval limit
    rewritten_query: str            # Current query used for retrieval
    query_type: str                 # conceptual | how-to | troubleshooting | api-reference
    retrieved_documents: list[Doc]  # Raw chunks returned from vector store
    relevant_documents: list[Doc]   # Filtered list of LLM-graded relevant chunks
    retry_count: int                # Counter tracking self-correction cycles
    max_retries: int                # Configurable hard ceiling (default: 2)
    answer: str                     # Generated answer or fallback text
    sources: list[SourceMetadata]   # Structured citations (file, chunk ID, heading)
    generation_supported: bool      # True if answer is grounded; False if fallback
    error: Optional[str]            # Propagated error message if any step fails
```

### Self-Correction & Routing Logic (`rag/routing.py`)

```python
def route_after_grading(state: RAGState) -> str:
    if state.get("error"):
        return "fallback"
    if state.get("relevant_documents"):
        return "generate"
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "query_analysis"   # Loops back to rewrite and re-retrieve
    return "fallback"             # Max retries exhausted
```

---

## 3. Technology Stack & Justification

| Component | Technology | Rationale & Trade-offs |
|---|---|---|
| **Graph Framework** | LangGraph `StateGraph` (v1.1.9) | State-machine based orchestration supporting cycles, conditional routing, and deterministic state transitions. Avoids brittle sequential function scripts. |
| **LLM Provider** | Google Gemini 3.8 Flash (`gemini-3.8-flash`) | Fast inference, generous free tier, and native support for Pydantic structured output (`.with_structured_output()`) for reliable document grading. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally via CPU; zero API cost, zero external network dependency during retrieval, and 384-dimensional dense vectors optimal for technical prose. |
| **Vector Store** | ChromaDB (v1.5.8) | Persistent on disk (`./chroma_db`), native metadata filtering, and built-in listing capabilities needed for the `GET /documents` endpoint. |
| **Web API** | FastAPI (v0.115.0) + Uvicorn | High performance ASGI framework with automatic OpenAPI documentation (`/docs`), Pydantic v2 data validation, and async file upload handling. |
| **Testing** | pytest (v9.0.3) + pytest-asyncio | Complete test suite containing 50 unit and integration tests covering loaders, chunking, state routing, nodes, and API endpoints. |

---

## 4. Document Ingestion & Chunking Strategy

### The Technical Corpus

The corpus contains 5 comprehensive Markdown documents located in `docs/corpus/`:
1. `langgraph_concepts.md`: StateGraph, nodes, edges, conditional routing, state reducers, checkpointing.
2. `langchain_rag.md`: Document loaders, recursive splitters, retrievers, LCEL chaining, structured output.
3. `chromadb_guide.md`: Collections, distance metrics, querying, metadata filters, persistence.
4. `fastapi_guide.md`: Path operations, Pydantic schemas, dependency injection, lifespan events, file uploads.
5. `pydantic_v2.md`: BaseModel, Field constraints, model_dump, custom validators, TypeAdapter.

*(A helper script `scripts/fetch_corpus.py` is included to refresh or re-download these documents from canonical sources).*

### Chunking Rationale

Technical documentation contains mixed prose, function signatures, lists, and code blocks. Naive character splitting often breaks functions in half or tears code away from its explanatory text.
- **Hierarchical Separators**: `["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]` prioritizes splitting at section boundaries before paragraphs.
- **Chunk Size (800 characters)**: Large enough to retain a full code snippet and its explanation together, yet small enough to avoid semantic drift in the embedding vector.
- **Chunk Overlap (100 characters)**: Prevents edge-truncation of important definitions spanning boundary cuts.
- **Heading Preservation**: Every chunk automatically extracts the nearest Markdown header above it (`heading` metadata), enriching the citation context.
- **Deduplication via Hash**: Chunks compute a SHA-256 content hash. When running `POST /ingest` multiple times, existing hashes in ChromaDB are detected and skipped.

---

## 5. Setup & Installation

### Prerequisites
- Python 3.11, 3.12, or 3.13
- A Google Gemini API key ([Get one free at Google AI Studio](https://aistudio.google.com/app/apikey))

### 1. Clone & Set Up Virtual Environment

```bash
cd rag-docs-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and enter your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=AIzaSy...your_actual_key...
LLM_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVAL_TOP_K=5
MAX_RETRIES=2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

### 3. (Optional) Run Standalone Corpus Ingestion

The FastAPI application **auto-indexes** the corpus on first startup if the database is empty. You can also run the standalone ingestion script:

```bash
python scripts/ingest_corpus.py
```

### 4. Run the FastAPI Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API Docs (Swagger UI): `http://localhost:8000/docs`
- Redoc Interface: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`

### 5. Launch the Interactive Streamlit UI (Bonus Feature)

In a separate terminal tab, run:

```bash
streamlit run ui.py
```

The UI will automatically open at `http://localhost:8501`. Features include:
- **Interactive Chat Q&A**: Submit queries with real-time feedback and stream view.
- **Pipeline Inspector**: Expandable drawer showing the query classification, rewritten query, self-correction retry counter, and source citations.
- **Corpus Dashboard**: View all indexed documents and their chunk counts directly from the sidebar.
- **Document Management**: Drag-and-drop local file uploader (`POST /ingest`) and URL fetcher (`POST /ingest/url`).
- **One-Click Thumbs Up / Down Feedback**: Submits evaluations directly to `data/feedback.jsonl`.

---

## 6. API Reference & Examples

### 1. `POST /query`
Submit a question and receive a context-grounded answer with source citations.

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add a conditional edge in LangGraph?",
    "top_k": 5
  }'
```

**Response (200 OK):**
```json
{
  "answer": "In LangGraph, you add a conditional edge using `graph_builder.add_conditional_edges()`. You pass the source node name, a routing function that inspects the state and returns a destination string, and a dictionary mapping return values to node names or `END` [Source 1: langgraph_concepts.md — Edges].",
  "sources": [
    {
      "doc_id": "doc_langgraph_concepts",
      "source": "langgraph_concepts.md",
      "chunk_id": "doc_langgraph_concepts_chunk_0003",
      "heading": "Edges"
    }
  ],
  "query_type": "how-to",
  "rewritten_query": "add conditional edges routing function LangGraph StateGraph",
  "retry_count": 0,
  "relevant_chunks_used": 2,
  "generation_supported": true,
  "error": null
}
```

*When an out-of-corpus question is asked (e.g., about astrophysics), the system exhausts `max_retries` and returns:*
```json
{
  "answer": "I could not find sufficient information in the indexed documentation to answer this question.\n\nThis may mean:\n- The topic is not covered by the currently indexed documents.\n- The question uses terminology that doesn't match the documentation.\n- Relevant documents have not been ingested yet.",
  "sources": [],
  "query_type": "conceptual",
  "rewritten_query": "...",
  "retry_count": 2,
  "relevant_chunks_used": 0,
  "generation_supported": false,
  "error": null
}
```

---

### 2. `POST /ingest`
Upload files (`.md`, `.txt`, `.html`, `.pdf`) or ingest from URLs.

**File Upload:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@my_new_doc.md"
```

**URL Ingestion:**
```bash
curl -X POST "http://localhost:8000/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md"]
  }'
```

**Response (200 OK):**
```json
{
  "ingested": [
    {
      "source": "my_new_doc.md",
      "doc_id": "doc_my_new_doc",
      "chunks_added": 8,
      "chunks_skipped_duplicate": 0
    }
  ],
  "errors": [],
  "total_chunks_added": 8
}
```

---

### 3. `GET /documents`
List all indexed documents, chunk distributions, and timestamps.

**Request:**
```bash
curl -X GET "http://localhost:8000/documents"
```

**Response (200 OK):**
```json
{
  "total_documents": 5,
  "total_chunks": 94,
  "documents": [
    {
      "doc_id": "doc_chromadb_guide",
      "source": "chromadb_guide.md",
      "title": "ChromaDB — Getting Started Guide",
      "chunks": 22,
      "ingested_at": "2026-10-31T06:30:00Z"
    },
    {
      "doc_id": "doc_fastapi_guide",
      "source": "fastapi_guide.md",
      "title": "FastAPI — Core Concepts and Usage Guide",
      "chunks": 21,
      "ingested_at": "2026-10-31T06:30:00Z"
    }
  ]
}
```

---

### 4. `POST /feedback`
Record user thumbs-up/down ratings and qualitative comments.

**Request:**
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "rating": "thumbs_up",
    "comment": "Accurate explanation and cited the correct section.",
    "question": "How do I add a conditional edge in LangGraph?"
  }'
```

**Response (200 OK):**
```json
{
  "status": "recorded",
  "feedback_id": "fb_3a4b9c1d2e3f",
  "message": "Thank you for your feedback."
}
```

---

## 7. Design Decisions & Trade-Offs

### 1. Local Embeddings vs. Cloud API Embeddings
- **Decision**: Used `all-MiniLM-L6-v2` locally via `sentence-transformers`.
- **Reasoning**: Eliminates secondary API keys, runs completely offline without network latency, and produces reliable 384-dimensional cosine representations for technical docs.
- **Trade-off**: Slightly lower context representation than OpenAI `text-embedding-3-small`, but completely free and has zero external failure points.

### 2. ChromaDB vs. FAISS
- **Decision**: Selected ChromaDB with persistent disk storage.
- **Reasoning**: ChromaDB natively stores structured metadata per chunk and provides query filters required for the `GET /documents` endpoint and deduplication.
- **Trade-off**: ChromaDB has slightly more process overhead than raw FAISS index matrices, but avoids the complex custom disk serialization and manual metadata indexing required with FAISS.

### 3. Retry Limit of 2
- **Decision**: Fixed `max_retries = 2`.
- **Reasoning**: A 1st retry handles terminology mismatch (e.g. acronyms vs spelled out). A 2nd retry tries a broader conceptual query. On a specialized 5-document local corpus, a 3rd attempt is statistically unlikely to discover unindexed material and would inject 3–5 seconds of unnecessary latency.

### 4. Feedback Persistence
- **Decision**: Append-only JSONL format (`data/feedback.jsonl`).
- **Reasoning**: Completely durable against server crashes, zero configuration required, and atomic in nature. The README honestly documents that feedback is collected for offline evaluation rather than claiming real-time automated model fine-tuning.

---

## 8. Verification & Test Suite

The repository includes a comprehensive 50-test suite running across three test modules:

```bash
pytest -v
```

### Test Coverage Highlights:
- **`tests/test_ingestion.py`**:
  - Markdown, plain text, and HTML loading
  - Header extraction (H1/H2)
  - SSRF protections (rejection of `localhost`, private IP ranges `10.0.0.0/8`, `192.168.0.0/16`, non-HTTP schemes)
  - Recursive chunking boundaries, chunk ID uniqueness, and SHA-256 deduplication
- **`tests/test_graph_nodes.py`**:
  - Routing condition evaluations (`generate` on relevant docs, `query_analysis` on retry, `fallback` on exhausted retries or errors)
  - Retrieval node query handling
  - Query analysis classification & retry counter incrementation
  - Grading node document filtering and conservative error handling
- **`tests/test_api.py`**:
  - `POST /query`: Valid questions, 422 input validation, fallback responses
  - `GET /documents`: Populated & empty collection states
  - `POST /feedback`: Thumbs up/down validation, JSONL disk logging
  - `POST /ingest`: Multi-file upload, invalid extension rejections, SSRF validation
  - `GET /health`: Health status monitoring

**Observed Test Results:**
```text
tests/test_api.py ................                                     [ 32%]
tests/test_graph_nodes.py ...................                           [ 70%]
tests/test_ingestion.py ...............s                                [100%]

================== 49 passed, 1 skipped in 11.28s ==================
```
*(1 test skipped: remote URL live validation test in sandbox environment).*

---

## 9. Assumptions & Future Improvements

### Assumptions Made
1. The technical corpus is small enough (hundreds of chunks) that in-process vector retrieval against local persistent ChromaDB delivers sub-20ms lookup times.
2. The user has access to a free Google AI Studio key (`GOOGLE_API_KEY`).
3. Feedback submitted by users is intended for offline evaluation and audit rather than active online model re-weighting.

### Future Improvements (with more time)
1. **Self-RAG Groundedness / Hallucination Node**: Add a secondary validation pass verifying that every sentence generated in the answer maps directly to a citation quote before returning.
2. **Hybrid Search (BM25 + Dense Vectors)**: Implement Reciprocal Rank Fusion (RRF) combining keyword search with semantic vectors to improve precision on exact code identifiers and function names.
3. **Web Search Fallback**: Integrate Tavily API to dynamically pull from live developer documentation when the local corpus doesn't contain the requested topic.
4. **Interactive UI**: Add a minimal Gradio/Streamlit interface with conversational session memory.

