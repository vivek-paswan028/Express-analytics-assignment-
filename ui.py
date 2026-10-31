"""
ui.py
─────
Interactive Streamlit frontend for the RAG Technical Documentation Assistant.
Communicates with the FastAPI backend via standard REST endpoints:
  - POST /query     -> Ask questions with self-corrective LangGraph pipeline
  - GET  /documents -> List indexed corpus documents
  - POST /ingest    -> Upload local files (.md, .txt, .html, .pdf)
  - POST /ingest/url-> Ingest from URL with SSRF protection
  - POST /feedback  -> Thumbs up / down feedback submission
  - GET  /health    -> Live status check
"""
from __future__ import annotations

import os
import requests
import streamlit as st

# ── Configuration & Page Setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── API Helper Functions ─────────────────────────────────────────────────────
def fetch_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def fetch_documents():
    try:
        res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception:
        pass
    return []


def submit_query(question: str, top_k: int = 5):
    res = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    if res.status_code == 200:
        return res.json()
    else:
        err_msg = res.json().get("detail", res.text)
        raise RuntimeError(f"API Error ({res.status_code}): {err_msg}")


def submit_feedback(rating: str, comment: str, question: str):
    res = requests.post(
        f"{API_BASE_URL}/feedback",
        json={"rating": rating, "comment": comment, "question": question},
        timeout=10,
    )
    return res.status_code == 200


def upload_files(files):
    upload_list = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    res = requests.post(f"{API_BASE_URL}/ingest", files=upload_list, timeout=60)
    if res.status_code == 200:
        return res.json()
    raise RuntimeError(res.json().get("detail", res.text))


def ingest_url(url: str):
    res = requests.post(
        f"{API_BASE_URL}/ingest/url",
        json={"urls": [url]},
        timeout=30,
    )
    if res.status_code == 200:
        return res.json()
    raise RuntimeError(res.json().get("detail", res.text))


# ── Sidebar: Knowledge Base & Management ─────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Control")
    health = fetch_health()

    if health:
        st.success(f"🟢 **API Online** (`{health.get('llm_model', 'N/A')}`)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", health.get("indexed_documents", 0))
        with col2:
            st.metric("Total Chunks", health.get("indexed_chunks", 0))
    else:
        st.error(f"🔴 **FastAPI Offline**\nCannot connect to `{API_BASE_URL}`.\nRun: `uvicorn app.main:app --reload`")

    st.markdown("---")
    st.subheader("📚 Indexed Documents")
    docs = fetch_documents()
    if docs:
        for d in docs:
            st.markdown(f"- **{d['source']}**  \n  `{d['chunks']} chunks` | `{d['doc_id']}`")
    else:
        st.info("No documents currently indexed.")

    st.markdown("---")
    st.subheader("📥 Ingestion Tools")

    # File uploader tab
    with st.expander("Upload Local Files"):
        uploaded_files = st.file_uploader(
            "Upload (.md, .txt, .html, .pdf)",
            type=["md", "txt", "html", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest Files", use_container_width=True):
            with st.spinner("Chunking & indexing into ChromaDB..."):
                try:
                    result = upload_files(uploaded_files)
                    st.success(f"Added {result.get('total_chunks_added', 0)} chunks!")
                    if result.get("errors"):
                        for e in result["errors"]:
                            st.warning(f"{e['source']}: {e['error']}")
                    st.rerun()
                except Exception as ex:
                    st.error(str(ex))

    # URL fetch tab
    with st.expander("Ingest from URL"):
        input_url = st.text_input("Raw Markdown or HTML URL")
        if input_url and st.button("Fetch & Ingest URL", use_container_width=True):
            with st.spinner("Fetching URL with SSRF checks..."):
                try:
                    result = ingest_url(input_url)
                    st.success(f"Added {result.get('total_chunks_added', 0)} chunks!")
                    if result.get("errors"):
                        for e in result["errors"]:
                            st.warning(f"{e['source']}: {e['error']}")
                    st.rerun()
                except Exception as ex:
                    st.error(str(ex))


# ── Main Q&A Interface ───────────────────────────────────────────────────────
st.title("🤖 Technical Documentation Assistant")
st.caption("Powered by LangGraph Self-Corrective Workflow, ChromaDB & FastAPI")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested Sample Questions
st.markdown("**Quick Prompts:**")
sample_cols = st.columns(4)
sample_prompts = [
    "How do I add a conditional edge in LangGraph?",
    "What is a StateGraph and how do nodes work?",
    "How does ChromaDB persist embeddings to disk?",
    "Explain quantum superposition in physics.",
]

selected_prompt = None
for i, prompt in enumerate(sample_prompts):
    with sample_cols[i]:
        if st.button(prompt, key=f"sample_{i}", use_container_width=True):
            selected_prompt = prompt

# Render Existing Messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]

            # Citations and pipeline drawer
            with st.expander("🔍 Pipeline Trace & Citations"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.caption(f"**Query Type:** `{meta.get('query_type')}`")
                with col_b:
                    st.caption(f"**Retries Used:** `{meta.get('retry_count')}`")
                with col_c:
                    status = "✅ Grounded" if meta.get("generation_supported") else "⚠️ Fallback"
                    st.caption(f"**Status:** {status}")

                st.markdown(f"**Rewritten Query:** `{meta.get('rewritten_query')}`")

                sources = meta.get("sources", [])
                if sources:
                    st.markdown("**Citations:**")
                    for s in sources:
                        heading = f" — *{s['heading']}*" if s.get("heading") else ""
                        st.markdown(f"- 📄 `{s['source']}`{heading} (`{s['chunk_id']}`)")
                else:
                    st.info("No source chunks utilized (fallback path triggered).")

            # Feedback component
            if f"feedback_{idx}" not in st.session_state:
                fb_col1, fb_col2 = st.columns([1, 4])
                with fb_col1:
                    rating = st.feedback("thumbs", key=f"fb_widget_{idx}")
                    if rating is not None:
                        rating_str = "thumbs_up" if rating == 1 else "thumbs_down"
                        if submit_feedback(rating_str, "", message.get("question", "")):
                            st.session_state[f"feedback_{idx}"] = rating_str
                            st.toast("Feedback recorded! Stored in data/feedback.jsonl", icon="✅")
                            st.rerun()

# Handle New User Input
user_query = st.chat_input("Ask a question about the indexed documentation...") or selected_prompt

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Invoke FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, retrieving chunks & grading relevance..."):
            try:
                response = submit_query(user_query)
                answer = response.get("answer", "No answer returned.")
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "question": user_query,
                    "metadata": {
                        "query_type": response.get("query_type"),
                        "rewritten_query": response.get("rewritten_query"),
                        "retry_count": response.get("retry_count"),
                        "sources": response.get("sources", []),
                        "generation_supported": response.get("generation_supported"),
                    },
                })
                st.rerun()

            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")
