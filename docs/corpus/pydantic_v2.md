# Pydantic V2 — Data Validation and Settings

> Source: Pydantic V2 official documentation (https://docs.pydantic.dev/latest/)
> License: MIT

## Overview

Pydantic is the most widely used Python data validation library. It uses Python type annotations to define schemas and validates data at runtime. Pydantic V2 (released 2023) rewrote the core in Rust for significantly better performance.

Key uses in FastAPI and LangChain applications:
- Request/response schema validation in FastAPI
- Structured LLM output with `with_structured_output()`
- Configuration management with `pydantic-settings`

## BaseModel

The core class. Define a model by inheriting from `BaseModel`:

```python
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None  # optional with default None
```

Pydantic validates input data on instantiation:

```python
user = User(id=1, name="Alice", email="alice@example.com")
print(user.name)  # "Alice"
print(user.model_dump())  # {"id": 1, "name": "Alice", "email": "alice@example.com", "age": None}
```

Invalid input raises `ValidationError`:
```python
User(id="not-an-int", name="Bob", email="bob@example.com")
# ValidationError: id: Input should be a valid integer
```

## Field

`Field()` adds constraints and metadata to model fields:

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,                   # ... means required (no default)
        min_length=1,
        max_length=2000,
        description="The user's question to answer",
        examples=["How do I add a node in LangGraph?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,            # greater than or equal to 1
        le=20,           # less than or equal to 20
        description="Number of chunks to retrieve",
    )
```

Common constraints:
- `min_length` / `max_length` — string length
- `ge` / `le` / `gt` / `lt` — numeric bounds
- `pattern` — regex pattern for strings
- `default` — default value (use `default_factory` for mutable defaults)

## Validators

### Field Validators (V2 syntax)

```python
from pydantic import BaseModel, field_validator

class FeedbackRequest(BaseModel):
    rating: str
    comment: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: str) -> str:
        allowed = {"thumbs_up", "thumbs_down"}
        if v not in allowed:
            raise ValueError(f"rating must be one of {allowed}, got '{v}'")
        return v
```

### Model Validators

Run validation after all fields are set:

```python
from pydantic import model_validator

class DateRange(BaseModel):
    start: int
    end: int

    @model_validator(mode="after")
    def check_range(self) -> "DateRange":
        if self.start > self.end:
            raise ValueError("start must be before end")
        return self
```

## TypedDict vs BaseModel

LangGraph state uses `TypedDict` (from Python's `typing` module), not Pydantic `BaseModel`. The difference:

| Feature | TypedDict | BaseModel |
|---------|-----------|-----------|
| Runtime validation | ❌ No | ✅ Yes |
| JSON serialization | Manual | `.model_dump()` / `.model_dump_json()` |
| LangGraph compatibility | ✅ Yes | ✅ Yes (with `from_attributes=True`) |
| Performance | Faster (just a dict) | Slower but safer |

Use `TypedDict` for LangGraph state (performance, LangGraph integration) and `BaseModel` for FastAPI request/response schemas (validation, serialization, docs).

## Literal Types for Enums

Use `Literal` to restrict a field to specific string values:

```python
from typing import Literal

class GradeDocument(BaseModel):
    score: Literal["relevant", "irrelevant"]
    reasoning: str

# LLM structured output with forced choices:
grader = llm.with_structured_output(GradeDocument)
result = grader.invoke([HumanMessage(content="Grade this document...")])
print(result.score)  # always "relevant" or "irrelevant"
```

`Literal` is preferred over `str` for constrained choices because:
1. Pydantic validates the value is in the allowed set
2. IDEs can provide autocomplete
3. The JSON schema includes the enum values, improving LLM structured output quality

## model_config

Class-level configuration for Pydantic models:

```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,     # strip whitespace from strings
        str_to_lower=False,
        populate_by_name=True,         # allow using field name as alias
        from_attributes=True,          # allow creating from ORM objects
        json_schema_extra={
            "example": {"name": "Alice"}
        },
    )

    name: str
```

## Pydantic Settings

`pydantic-settings` extends Pydantic to load configuration from environment variables and `.env` files:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    retrieval_top_k: int = 5

settings = Settings()
print(settings.google_api_key)  # loaded from GOOGLE_API_KEY env var or .env
```

Loading priority (highest first):
1. Environment variables (override everything)
2. `.env` file values
3. Default values in the model

## Serialization

```python
model = QueryResponse(answer="...", sources=[...], retry_count=0)

# To Python dict
d = model.model_dump()

# To JSON string
j = model.model_dump_json()

# To JSON string with indentation
j_pretty = model.model_dump_json(indent=2)

# Exclude fields
d = model.model_dump(exclude={"internal_field"})

# Include only specific fields
d = model.model_dump(include={"answer", "sources"})
```

## Validation Without Instantiation

```python
from pydantic import TypeAdapter

# Validate a raw value against a type
adapter = TypeAdapter(list[int])
result = adapter.validate_python([1, 2, "3"])  # "3" coerces to 3
```

## Common Patterns in FastAPI+LangChain

### Source Citation Model

```python
class SourceMetadata(BaseModel):
    doc_id: str
    source: str          # filename or URL
    chunk_id: str
    heading: str = ""    # nearest section heading

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceMetadata]
    query_type: str
    rewritten_query: str
    retry_count: int
    relevant_chunks_used: int
```

### Structured LLM Grading

```python
class DocumentGrade(BaseModel):
    """Grade a single retrieved document chunk."""
    score: Literal["relevant", "irrelevant"]
    reasoning: str = Field(description="One-sentence explanation of the grade")

# Use with LangChain
grader_llm = llm.with_structured_output(DocumentGrade)
grade: DocumentGrade = grader_llm.invoke(messages)
if grade.score == "relevant":
    relevant_docs.append(doc)
```

