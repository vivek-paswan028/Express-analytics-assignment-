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

_GENERATION_SYSTEM_PROMPT = """You are a precise technical documentation assistant.

STRICT RULES:
1. Answer using ONLY the information in the provided context sections.
2. Do NOT use any knowledge from your training that is not present in the context.
3. If the context does not fully answer the question, say so explicitly and answer only what the context supports.
4. Cite sources inline using the format [Source N] where N is the source number.
5. Be concise and technical — this is a documentation assistant, not a chatbot.
6. Use code blocks (```python) for any code examples you include.
7. If you are uncertain about any claim because the context is ambiguous, say "According to the documentation..." or "The context suggests..." rather than stating it as fact.
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
        f"Context from technical documentation:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer the question based strictly on the context above. "
        f"Use [Source N] citations inline."
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

