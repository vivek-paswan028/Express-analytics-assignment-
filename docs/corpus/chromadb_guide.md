# ChromaDB — Getting Started Guide

> Source: ChromaDB official documentation (https://docs.trychroma.com/)
> License: Apache 2.0

## Overview

ChromaDB is an open-source AI-native vector database designed for storing and querying embeddings. It is the recommended vector store for prototyping and small-to-medium RAG applications because it:

- Runs entirely locally with no external services required
- Persists data to disk automatically
- Supports metadata filtering alongside vector search
- Has a simple Python API
- Is free and open-source

## Installation

```bash
pip install chromadb
```

## Basic Usage

### Creating a Client

ChromaDB supports two client modes:

**Ephemeral (in-memory, not persisted):**
```python
import chromadb

client = chromadb.Client()
```

**Persistent (saved to disk):**
```python
client = chromadb.PersistentClient(path="./chroma_db")
```

The persistent client is strongly recommended for production use and applications that need to survive restarts. Data is stored in the directory specified by `path`.

### Collections

A collection is ChromaDB's equivalent of a table — a named group of embeddings with associated metadata and documents.

```python
# Create or get an existing collection
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"},  # use cosine similarity
)
```

The `hnsw:space` metadata controls the distance metric:
- `"cosine"` — cosine similarity (recommended for text embeddings)
- `"l2"` — Euclidean distance
- `"ip"` — inner product

### Adding Documents

```python
collection.add(
    documents=["LangGraph is a stateful workflow library...", "FastAPI is a web framework..."],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # pre-computed embeddings
    metadatas=[
        {"source": "langgraph.md", "doc_id": "doc_001", "chunk_id": "chunk_001"},
        {"source": "fastapi.md", "doc_id": "doc_002", "chunk_id": "chunk_002"},
    ],
    ids=["chunk_001", "chunk_002"],  # unique IDs per document
)
```

**Important:** IDs must be unique within a collection. Attempting to add a document with a duplicate ID raises an error. Use the `upsert` method to add-or-update.

### Querying

```python
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],  # embedding of the user's query
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
```

The result is a dictionary with parallel lists:
```python
{
    "ids": [["chunk_003", "chunk_007", ...]],
    "documents": [["LangGraph supports cycles...", "To add an edge..."]],
    "metadatas": [[{"source": "langgraph.md", ...}, ...]],
    "distances": [[0.12, 0.18, ...]],   # lower = more similar (cosine)
}
```

### Filtering with Metadata

ChromaDB supports `where` clauses for metadata filtering:

```python
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"doc_id": "doc_001"},  # only search within this document
)
```

Complex filters using `$and`, `$or`, `$in`:
```python
where={
    "$and": [
        {"doc_id": {"$in": ["doc_001", "doc_002"]}},
        {"content_type": "prose"},
    ]
}
```

### Listing and Counting

```python
# Count all items in a collection
count = collection.count()

# Get all items (use carefully on large collections)
all_items = collection.get(include=["metadatas"])

# Get items by ID
specific = collection.get(ids=["chunk_001", "chunk_002"])

# Get with metadata filter
filtered = collection.get(where={"doc_id": "doc_001"})
```

### Updating and Deleting

```python
# Update existing documents
collection.update(
    ids=["chunk_001"],
    documents=["Updated content..."],
    embeddings=[[0.2, 0.3, ...]],
    metadatas=[{"source": "updated.md"}],
)

# Upsert (add if not exists, update if exists)
collection.upsert(
    ids=["chunk_001"],
    documents=["..."],
    embeddings=[[...]],
    metadatas=[{...}],
)

# Delete by ID
collection.delete(ids=["chunk_001"])

# Delete by metadata filter
collection.delete(where={"doc_id": "doc_old"})
```

### Listing Collections

```python
# List all collection names
collections = client.list_collections()
# Returns a list of Collection objects

# Delete a collection
client.delete_collection("my_documents")

# Reset (delete all collections) — use with caution
client.reset()
```

## Integration with LangChain

LangChain provides a `Chroma` wrapper that handles embedding generation automatically:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create from documents (ingestion)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_documents",
)

# Load existing (retrieval)
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_documents",
)

# Query
docs = vector_store.similarity_search("my question", k=5)
docs_with_scores = vector_store.similarity_search_with_score("my question", k=5)
```

## Performance Considerations

ChromaDB uses [HNSW](https://arxiv.org/abs/1603.09320) (Hierarchical Navigable Small World) for approximate nearest neighbour search. HNSW performance characteristics:

- Query time: O(log n) — very fast even with millions of vectors
- Index build time: O(n log n)
- Memory: higher than FAISS flat index, but HNSW is pre-built and doesn't require GPU

For collections under ~100,000 vectors (typical for documentation RAG), ChromaDB's performance is more than sufficient. FAISS becomes worth considering at millions of vectors with strict latency requirements.

## Deduplication Strategy

To avoid ingesting the same chunk twice, store a content hash in the metadata:

```python
import hashlib

def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

# Before adding, check if the hash already exists
existing = collection.get(where={"content_hash": chunk_hash(chunk_text)})
if existing["ids"]:
    print(f"Skipping duplicate chunk")
else:
    collection.add(...)
```

## Backup and Migration

The persistent ChromaDB directory contains SQLite files and binary index files. To back up or move a database, copy the entire directory. To start fresh, delete the directory.

```bash
# Backup
cp -r ./chroma_db ./chroma_db_backup

# Reset (removes all indexed data)
rm -rf ./chroma_db
```

