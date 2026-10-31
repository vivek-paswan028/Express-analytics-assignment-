# LangChain RAG (Retrieval-Augmented Generation)

> Source: LangChain official documentation (https://python.langchain.com/docs/tutorials/rag/)
> License: MIT

## Overview

Retrieval-Augmented Generation (RAG) is a technique that grounds LLM responses in external knowledge. Instead of relying solely on the model's pretrained weights, RAG retrieves relevant documents from a knowledge base and passes them as context to the LLM at inference time.

A RAG application has two main components:
1. **Indexing** — loading, chunking, embedding, and storing documents
2. **Retrieval and Generation** — retrieving relevant chunks, then generating an answer

## Indexing Pipeline

### Step 1: Load Documents

LangChain provides `DocumentLoader` classes for many formats:

```python
from langchain_community.document_loaders import TextLoader, WebBaseLoader

# Load from a local file
loader = TextLoader("my_document.txt")
docs = loader.load()

# Load from a URL
web_loader = WebBaseLoader("https://example.com/docs")
web_docs = web_loader.load()
```

Each document is a `Document` object with two fields:
- `page_content: str` — the text content
- `metadata: dict` — arbitrary key-value pairs (source, page number, etc.)

### Step 2: Split into Chunks

Documents are split into smaller chunks for two reasons:
1. Embedding models have token limits
2. Smaller chunks have more focused semantic meaning, improving retrieval precision

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)

chunks = splitter.split_documents(docs)
```

`RecursiveCharacterTextSplitter` splits on the given separators in order, falling back to the next separator if the current chunk is still too large. This keeps semantic units (paragraphs, sections) together better than naive fixed-size splitting.

### Step 3: Embed and Store

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="rag_documents",
)
```

## Retrieval

### Similarity Search

The simplest retrieval method is cosine similarity search against the embedding space:

```python
results = vector_store.similarity_search(query="How do I add a node?", k=5)
```

Returns the top-k most similar `Document` objects.

### Similarity Search with Score

```python
results_with_scores = vector_store.similarity_search_with_score(query, k=5)
# Returns List[Tuple[Document, float]]
# Score is cosine distance (lower = more similar in Chroma)
```

### Retrievers

Retrievers are a higher-level abstraction. Convert any vector store to a retriever:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("my question")
```

Retrievers compose easily with LCEL chains:

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)
```

## Document Object

Every document has:
```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangGraph is a library for building stateful workflows...",
    metadata={
        "source": "langgraph_concepts.md",
        "doc_id": "doc_001",
        "chunk_id": "doc_001_chunk_003",
        "heading": "Key Concepts",
    }
)
```

The `metadata` dict is preserved through retrieval and is critical for citation generation.

## Generation with Context

### Prompt Construction

```python
from langchain_core.prompts import ChatPromptTemplate

template = """You are a technical documentation assistant.
Answer the question using ONLY the provided context.
If the context does not contain enough information, say so explicitly.

Context:
{context}

Question: {question}

Answer with citations referencing the source documents."""

prompt = ChatPromptTemplate.from_template(template)
```

### Formatting Retrieved Documents

```python
def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        heading = doc.metadata.get("heading", "")
        parts.append(f"[Source {i}: {source} — {heading}]\n{doc.page_content}")
    return "\n\n".join(parts)
```

## LangChain Expression Language (LCEL)

LCEL uses the pipe operator `|` to chain Runnables:

```python
chain = prompt | llm | output_parser
result = chain.invoke({"context": context_str, "question": "..."})
```

Every component in the chain implements the `Runnable` interface with `.invoke()`, `.stream()`, and `.batch()` methods.

### Structured Output

Force an LLM to return a structured Pydantic model:

```python
from pydantic import BaseModel

class GradeDocument(BaseModel):
    score: str  # "relevant" or "irrelevant"
    reasoning: str

structured_llm = llm.with_structured_output(GradeDocument)
result: GradeDocument = structured_llm.invoke(messages)
```

`with_structured_output` uses function calling / tool use under the hood to guarantee the output matches the schema.

## Self-Corrective RAG

The key insight of self-corrective RAG is that retrieval quality can be poor — the embedding model may not surface the best chunks for every question. The solution is to:

1. Grade retrieved documents with an LLM
2. If quality is insufficient, rewrite the query and retrieve again
3. Track retry counts to avoid infinite loops

### Query Rewriting

```python
rewrite_prompt = ChatPromptTemplate.from_template("""
You are improving a search query for a technical documentation retrieval system.

Original question: {question}
Retrieval attempt: {retry_count}

Generate an improved search query that:
- Uses different technical terminology
- Is more specific
- Avoids ambiguous pronouns
- Focuses on the core technical concept

Return only the improved query, nothing else.
""")

rewriter = rewrite_prompt | llm | StrOutputParser()
new_query = rewriter.invoke({"question": question, "retry_count": 1})
```

### Document Grading

```python
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are grading whether a document chunk is relevant to a question.
    Grade as 'relevant' if the chunk contains information that helps answer the question.
    Grade as 'irrelevant' if the chunk is off-topic or does not help.
    Be conservative — only grade as relevant if there is clear topical overlap."""),
    ("human", "Question: {question}\n\nDocument chunk:\n{document}\n\nGrade:"),
])

grader = grade_prompt | structured_llm
```

## Hallucination and Grounding

A grounding check verifies that the generated answer is supported by the retrieved context. This prevents the LLM from injecting information from its pretrained knowledge that wasn't in the retrieved documents.

```python
class GroundingCheck(BaseModel):
    grounded: bool
    unsupported_claims: list[str]

grounding_checker = grounding_prompt | llm.with_structured_output(GroundingCheck)
```

If `grounded` is `False`, the system can either regenerate the answer (prompting the LLM to stay on-context) or flag the answer as potentially unreliable.

