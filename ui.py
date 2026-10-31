"""
ui.py
─────
Interactive Streamlit frontend for the RAG Technical Documentation Assistant.
Features:
  - Tab 1: 💬 Q&A Chat Assistant with pipeline trace & citations
  - Tab 2: 📁 Document Management (Upload files, Ingest from URL, View catalog, Clear database)
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

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# ── API Helper Functions ─────────────────────────────────────────────────────
def fetch_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    # Try localhost fallback if 127.0.0.1 didn't respond
    if "127.0.0.1" in API_BASE_URL:
        try:
            res = requests.get(f"{API_BASE_URL.replace('127.0.0.1', 'localhost')}/health", timeout=3)
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


def clear_all_documents():
    try:
        res = requests.delete(f"{API_BASE_URL}/documents", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


def delete_single_document(doc_id: str):
    try:
        res = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


def submit_query(question: str, top_k: int = 5):
    res = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    try:
        data = res.json()
    except Exception:
        data = {"detail": res.text}
    if res.status_code == 200:
        return data
    else:
        err_msg = data.get("detail", res.text)
        raise RuntimeError(f"API Error ({res.status_code}): {err_msg}")


def submit_feedback(rating: str, comment: str, question: str):
    try:
        res = requests.post(
            f"{API_BASE_URL}/feedback",
            json={"rating": rating, "comment": comment, "question": question},
            timeout=10,
        )
        return res.status_code == 200
    except Exception:
        return False


def upload_files(files):
    upload_list = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    res = requests.post(f"{API_BASE_URL}/ingest", files=upload_list, timeout=120)
    try:
        data = res.json()
    except Exception:
        data = {"detail": res.text}
    if res.status_code == 200:
        return data
    raise RuntimeError(f"Upload failed ({res.status_code}): {data.get('detail', res.text)}")


def ingest_url(url: str):
    res = requests.post(
        f"{API_BASE_URL}/ingest/url",
        json={"urls": [url]},
        timeout=30,
    )
    try:
        data = res.json()
    except Exception:
        data = {"detail": res.text}
    if res.status_code == 200:
        return data
    raise RuntimeError(f"URL ingest failed ({res.status_code}): {data.get('detail', res.text)}")


# ── Sidebar: System Health & Navigation ───────────────────────────────────────
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
    page_nav = st.radio(
        "Navigation",
        ["💬 Ask Questions (Chat)", "📁 Manage & Ingest Documents"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("📚 Quick Document List")
    docs = fetch_documents()
    if docs:
        for d in docs:
            st.markdown(f"• **{d['source']}**  \n  `{d['chunks']} chunks` (`{d['doc_id']}`)")
        st.markdown("")
        if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
            if clear_all_documents():
                st.toast("Cleared all documents from vector store!", icon="🗑️")
                st.rerun()
            else:
                st.error("Failed to clear documents.")
    else:
        st.info("No documents currently indexed.")


# ── Main Content Area ────────────────────────────────────────────────────────
st.title("🤖 Technical Documentation Assistant")
st.caption("Self-corrective RAG with LangGraph, ChromaDB, and FastAPI")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1: Chat Q&A
# ─────────────────────────────────────────────────────────────────────────────
if page_nav == "💬 Ask Questions (Chat)":
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sample Questions
    st.markdown("**Suggested Questions:**")
    sample_cols = st.columns(4)
    sample_prompts = [
        "How do I add a conditional edge in LangGraph?",
        "What is a StateGraph and how do nodes work?",
        "How does ChromaDB persist embeddings to disk?",
        "Explain quantum entanglement in physics.",
    ]

    selected_prompt = None
    for i, prompt in enumerate(sample_prompts):
        with sample_cols[i]:
            if st.button(prompt, key=f"sample_{i}", use_container_width=True):
                selected_prompt = prompt

    st.markdown("---")

    # Render Chat History
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]

                # Expandable pipeline trace
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

                # Feedback widget
                if f"feedback_{idx}" not in st.session_state:
                    fb_col1, _ = st.columns([1, 4])
                    with fb_col1:
                        rating = st.feedback("thumbs", key=f"fb_widget_{idx}")
                        if rating is not None:
                            rating_str = "thumbs_up" if rating == 1 else "thumbs_down"
                            if submit_feedback(rating_str, "", message.get("question", "")):
                                st.session_state[f"feedback_{idx}"] = rating_str
                                st.toast("Feedback recorded to data/feedback.jsonl!", icon="✅")
                                st.rerun()

    # User Input
    user_query = st.chat_input("Ask a question about the indexed documentation...") or selected_prompt

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

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


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2: Document Management & Ingestion
# ─────────────────────────────────────────────────────────────────────────────
elif page_nav == "📁 Manage & Ingest Documents":
    st.subheader("📁 Ingest New Documentation")
    st.markdown("Upload documents or enter URLs to index them into ChromaDB.")

    upload_col, url_col = st.columns(2)

    with upload_col:
        st.markdown("#### 📄 Upload Local Files")
        uploaded_files = st.file_uploader(
            "Select documentation files (`.md`, `.txt`, `.html`, `.pdf`)",
            type=["md", "txt", "html", "pdf"],
            accept_multiple_files=True,
            key="file_uploader_widget",
        )

        if uploaded_files:
            st.caption(f"Selected {len(uploaded_files)} file(s): " + ", ".join([f.name for f in uploaded_files]))

            if st.button("🚀 Ingest Selected Files", type="primary", use_container_width=True):
                with st.spinner("Processing documents, splitting into chunks & generating embeddings..."):
                    try:
                        result = upload_files(uploaded_files)
                        added = result.get("total_chunks_added", 0)

                        if added > 0:
                            st.success(f"🎉 Successfully ingested! Added **{added} chunks** into ChromaDB.")
                        else:
                            st.warning("No new chunks added. (Files might already be indexed or contain duplicate content).")

                        # Display per-file breakdown
                        for item in result.get("ingested", []):
                            st.markdown(f"- **{item['source']}**: `{item['chunks_added']} added`, `{item['chunks_skipped_duplicate']} skipped (duplicates)`")

                        if result.get("errors"):
                            for err in result["errors"]:
                                st.error(f"❌ Error in {err['source']}: {err['error']}")

                    except Exception as ex:
                        st.error(f"Upload failed: {str(ex)}")

    with url_col:
        st.markdown("#### 🌐 Ingest from URL")
        input_url = st.text_input(
            "Enter raw document or web documentation URL",
            placeholder="https://raw.githubusercontent.com/.../README.md",
        )

        if st.button("📥 Fetch & Ingest URL", use_container_width=True):
            if not input_url:
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching URL and generating embeddings..."):
                    try:
                        result = ingest_url(input_url)
                        added = result.get("total_chunks_added", 0)
                        if added > 0:
                            st.success(f"🎉 Added **{added} chunks** from URL!")
                        else:
                            st.warning("No new chunks added (duplicate content).")

                        for err in result.get("errors", []):
                            st.error(f"❌ {err['source']}: {err['error']}")

                    except Exception as ex:
                        st.error(f"URL Ingestion failed: {str(ex)}")

    st.markdown("---")
    st.subheader("📋 Currently Indexed Documents")
    current_docs = fetch_documents()

    if current_docs:
        for doc in current_docs:
            d_col1, d_col2, d_col3 = st.columns([3, 1, 1])
            with d_col1:
                st.markdown(f"**{doc['source']}** (`{doc['doc_id']}`)")
                if doc.get("title"):
                    st.caption(f"Title: {doc['title']}")
            with d_col2:
                st.markdown(f"`{doc['chunks']} chunks`")
            with d_col3:
                if st.button("Delete", key=f"del_{doc['doc_id']}"):
                    if delete_single_document(doc["doc_id"]):
                        st.toast(f"Deleted {doc['source']}", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("Delete failed.")
            st.divider()

        if st.button("🗑️ Wipe All Documents", type="secondary"):
            if clear_all_documents():
                st.toast("Cleared all documents!", icon="🗑️")
                st.rerun()
            else:
                st.error("Failed to clear documents.")
    else:
        st.info("The vector store is currently empty. Use the upload box above to ingest your first document!")
