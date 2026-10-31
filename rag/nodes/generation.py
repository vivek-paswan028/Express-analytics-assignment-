from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from rag.llm import get_gemini_llm
from rag.state import RAGState, SourceMetadata

logger = logging.getLogger(__name__)

settings = get_settings()

_GENERATION_SYSTEM_PROMPT = """You are a senior technical documentation and knowledge assistant.
Your task is to provide a complete, well-structured, and comprehensive answer to the user's question based strictly on the provided context sections.

GUIDELINES FOR HIGH-QUALITY OUTPUT:
1. Thorough & Complete: Provide a detailed, full-fledged answer with complete meaning. Synthesize all relevant facts, names, technical specifications, skills, project overviews, steps, and explanations available in the context without truncating or leaving out important details.
2. Structure & Readability: Organize your answer logically using clear Markdown headings (##, ###), bullet points, bold emphasis, and numbered steps where appropriate.
3. Source Grounding & Citations: Attribute claims and facts using inline citations in the format [Source N] corresponding to the numbered context blocks.
4. Code Examples: Include formatted code blocks (```python, ```bash, etc.) whenever code, configuration, or commands are referenced in the context.
5. Honesty: Ground your response in the provided context. If a specific detail is absent, state what the context does contain and clearly explain any limitations.
"""


def _format_context(docs: list[Document]) -> tuple[str, list[SourceMetadata]]:
    parts: list[str] = []
    sources: list[SourceMetadata] = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        heading = meta.get("heading", "")
        doc_id = meta.get("doc_id", "")
        chunk_id = meta.get("chunk_id", "")

        label = f"{source} — {heading}" if heading else source
        parts.append(f"[Source {i}: {label}]\n{doc.page_content}")

        sources.append(SourceMetadata(
            doc_id=doc_id,
            source=source,
            chunk_id=chunk_id,
            heading=heading,
        ))

    return "\n\n---\n\n".join(parts), sources


def generation_node(state: RAGState) -> dict:
    question = state["original_question"]
    relevant_docs = state.get("relevant_documents", [])

    if not relevant_docs:
        logger.warning("generation_node called with no relevant documents.")
        return {
            "answer": (
                "I could not find sufficient information in the indexed documentation "
                "to answer this question confidently."
            ),
            "sources": [],
            "generation_supported": False,
        }

    context_str, sources = _format_context(relevant_docs)

    human_content = (
        f"Context from indexed documentation:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Provide a comprehensive, well-structured, and fully articulated answer based on the context above. "
        f"Include relevant names, definitions, specifications, and code examples. Cite your sources using [Source N] inline."
    )

    try:
        llm = get_gemini_llm(temperature=settings.llm_temperature)

        response = llm.invoke([
            SystemMessage(content=_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])

        raw_content = response.content
        if isinstance(raw_content, str):
            answer = raw_content.strip()
        elif isinstance(raw_content, list):
            parts = []
            for part in raw_content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
                elif hasattr(part, "text"):
                    parts.append(str(part.text))
            answer = "".join(parts).strip()
        else:
            answer = str(raw_content).strip()

        logger.info(
            "Generation complete: %d chars, %d sources cited",
            len(answer), len(sources),
        )

        return {
            "answer": answer,
            "sources": sources,
            "generation_supported": True,
            "error": None,
        }

    except Exception as e:
        logger.error("generation_node failed: %s", e)
        return {
            "answer": (
                "I encountered an error while generating the answer. "
                "Please check that your GOOGLE_API_KEY is set and try again."
            ),
            "sources": [],
            "generation_supported": False,
            "error": f"Generation failed: {str(e)}",
        }

