"""
rag/nodes/generation.py
────────────────────────
Node 4 of the LangGraph RAG workflow.

Generates the final answer using only the relevant, graded documents
as context. Includes citations in a consistent format.

Key constraints enforced in the prompt:
1. Answer ONLY from the provided context — do not use pretrained knowledge.
2. If context is incomplete, acknowledge uncertainty explicitly.
3. Every factual claim must be attributable to a specific source chunk.

Citation format: [Source N: filename — heading] inline in the answer,
with a full source list appended at the end.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
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
    """
    Format retrieved documents as numbered context sections.
    Returns the formatted context string and the source metadata list.
    """
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
    """
    Generate a grounded answer from the relevant documents.

    Returns state updates with:
    - answer: the generated answer string with inline citations
    - sources: list of SourceMetadata for the chunks used
    - generation_supported: True (we have relevant context)
    """
    question = state["original_question"]
    relevant_docs = state.get("relevant_documents", [])

    if not relevant_docs:
        # Should not reach here (routing prevents it), but handle defensively
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
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )

        response = llm.invoke([
            SystemMessage(content=_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])

        answer = response.content.strip()

        logger.info(
            "Generation complete: %d chars, %d sources cited",
            len(answer), len(sources),
        )

        return {
            "answer": answer,
            "sources": sources,
            "generation_supported": True,
            "error": None,  # clear any previous error
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

