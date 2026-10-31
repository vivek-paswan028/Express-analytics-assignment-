# FastAPI — Core Concepts and Usage Guide

> Source: FastAPI official documentation (https://fastapi.tiangolo.com/)
> License: MIT

## Overview

FastAPI is a modern, high-performance Python web framework for building APIs. Its key features are:

- **Fast to code** — type hints drive automatic validation, serialization, and documentation
- **Fast to run** — based on Starlette (ASGI) and Pydantic; comparable to NodeJS and Go in benchmarks
- **Automatic interactive docs** — Swagger UI at `/docs` and ReDoc at `/redoc`
- **Standards-based** — OpenAPI and JSON Schema

## Installation

```bash
pip install fastapi uvicorn[standard]
```

## First Application

```python
from fastapi import FastAPI

app = FastAPI(title="My API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Hello, World"}
```

Run with:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag watches for file changes. Remove it in production.

## Path Operations

Path operations are functions decorated with HTTP method decorators:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

- `item_id: int` — path parameter (FastAPI validates and converts type automatically)
- `q: str | None = None` — optional query parameter

Supported decorators: `@app.get`, `@app.post`, `@app.put`, `@app.patch`, `@app.delete`

## Request Body with Pydantic

Define request bodies using Pydantic models:

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

@app.post("/query")
def query(request: QueryRequest):
    return {"received": request.question}
```

FastAPI reads the body from JSON, validates it against the schema, and rejects invalid requests with a `422 Unprocessable Entity` response automatically.

## Response Models

Define explicit response schemas with `response_model`:

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceMetadata]
    retry_count: int

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    ...
```

Using `response_model` ensures the response is validated and documented in the OpenAPI spec. Fields not in the response model are automatically excluded.

## HTTP Status Codes

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# Raise an error
@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    return db[item_id]

# Custom status code on success
@app.post("/items", status_code=201)
def create_item(item: Item):
    db[item.id] = item
    return item
```

Common status codes:
- `200 OK` — default success
- `201 Created` — resource created
- `204 No Content` — success with no body
- `400 Bad Request` — client-side input error
- `404 Not Found` — resource doesn't exist
- `422 Unprocessable Entity` — validation failure (FastAPI generates these automatically)
- `500 Internal Server Error` — unexpected server error

## File Uploads

For file upload endpoints, use `UploadFile`:

```python
from fastapi import UploadFile, File
from typing import list

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
        })
    return results
```

`UploadFile` provides:
- `.filename` — original filename
- `.content_type` — MIME type
- `.read()` — async read entire file
- `.file` — underlying SpooledTemporaryFile

## Dependency Injection

FastAPI's `Depends` mechanism provides reusable, testable dependencies:

```python
from fastapi import Depends

def get_vector_store():
    # Returns the singleton ChromaDB client
    return vector_store_instance

@app.get("/documents")
def list_documents(store=Depends(get_vector_store)):
    return store.list_documents()
```

Dependencies can be nested, cached per-request, and overridden in tests:

```python
# In tests — override the dependency
app.dependency_overrides[get_vector_store] = lambda: mock_store
```

## Lifespan Events (startup/shutdown)

The recommended way to run initialisation code (loading models, warming up connections):

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs before the app accepts requests
    load_embedding_model()
    initialise_vector_store()
    compile_langgraph_workflow()
    yield
    # Shutdown: runs when the app is stopping
    cleanup_resources()

app = FastAPI(lifespan=lifespan)
```

## CORS Middleware

Allow browsers to call the API from a different origin:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Background Tasks

Run slow operations after returning the response:

```python
from fastapi import BackgroundTasks

def write_to_log(message: str):
    with open("log.txt", "a") as f:
        f.write(message + "\n")

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_to_log, f"Feedback: {feedback.rating}")
    return {"status": "recorded"}
```

## Testing FastAPI Applications

FastAPI includes a `TestClient` (wraps `httpx.Client`):

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_query():
    response = client.post("/query", json={"question": "What is LangGraph?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data

def test_validation():
    response = client.post("/query", json={"question": ""})  # empty question
    assert response.status_code == 422
```

## Path and Query Parameters Summary

```python
# Path parameter: /items/{item_id}
@app.get("/items/{item_id}")
def read(item_id: str):
    ...

# Query parameter: /search?q=foo&limit=10
@app.get("/search")
def search(q: str, limit: int = 10):
    ...

# Request body (POST/PUT): JSON in body
@app.post("/items")
def create(item: ItemModel):  # Pydantic model = body
    ...

# Mixed: path + body
@app.put("/items/{item_id}")
def update(item_id: str, item: ItemModel):
    ...
```

## Error Handling Best Practices

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )

# Never expose internal stack traces in production
# Use HTTPException with user-friendly messages
raise HTTPException(
    status_code=503,
    detail="LLM service temporarily unavailable. Please try again.",
)
```

