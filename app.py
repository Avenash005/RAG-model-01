"""Streamlit web application for the AI Campus Knowledge Assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from graph_core import execute_workflow
from rag_core import (
    DEFAULT_KB_DIR,
    extract_pdf_chunks,
    get_chroma_collection,
    get_embedding_model,
    index_knowledge_base,
    split_chunks,
)

load_dotenv()

st.set_page_config(
    page_title="Campus Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .route-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .route-docs {
        background-color: #1E3A8A;
        color: #93C5FD;
        border: 1px solid #3B82F6;
    }
    .route-tools {
        background-color: #064E3B;
        color: #6EE7B7;
        border: 1px solid #10B981;
    }
    .route-decline {
        background-color: #7F1D1D;
        color: #FCA5A5;
        border: 1px solid #EF4444;
    }
    .path-text {
        font-size: 0.75rem;
        color: #64748B;
        font-family: monospace;
        margin-bottom: 0.8rem;
    }
    .suggestion-btn {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #E2E8F0;
        text-align: left;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_cached_embedding_model():
    """Cache the sentence-transformers model instance across reruns."""
    return get_embedding_model()


# Initialize Knowledge Base once
@st.cache_resource(show_spinner=False)
def initialize_knowledge_base():
    """Ensure standard campus knowledge base documents are indexed on startup."""
    emb_model = load_cached_embedding_model()
    return index_knowledge_base(
        kb_dir=DEFAULT_KB_DIR,
        embedding_model=emb_model,
        force_reindex=False,
    )


total_kb_chunks = initialize_knowledge_base()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Sidebar Controls
with st.sidebar:
    st.title("🎓 Campus Assistant")
    st.caption("v1.0 • LU10 AI Capstone Project")

    st.markdown("---")
    st.subheader("🔑 API Key Configuration")

    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your Gemini API key from Google AI Studio. It is never stored or logged.",
        placeholder="AIzaSy...",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API Key configured", icon="✅")
    else:
        st.warning("Please provide a Gemini API key to query.", icon="⚠️")

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    st.write(f"Indexed Chunks: `{total_kb_chunks}`")

    if st.button("🔄 Re-index Knowledge Base", use_container_width=True):
        with st.spinner("Re-indexing knowledge base..."):
            total_kb_chunks = index_knowledge_base(
                kb_dir=DEFAULT_KB_DIR,
                embedding_model=load_cached_embedding_model(),
                force_reindex=True,
            )
        st.success(f"Successfully re-indexed {total_kb_chunks} chunks!")
        st.rerun()

    # Document upload for supplementary PDFs
    with st.expander("📄 Upload Supplementary PDF"):
        uploaded_pdf = st.file_uploader("Upload course PDF", type=["pdf"])
        if uploaded_pdf and st.button("Add PDF to Knowledge Base"):
            with st.spinner("Extracting and indexing PDF..."):
                raw_pdf_chunks = extract_pdf_chunks(uploaded_pdf.getvalue(), uploaded_pdf.name)
                split_pdf = split_chunks(raw_pdf_chunks)
                collection = get_chroma_collection()
                emb_model = load_cached_embedding_model()
                texts = [c.text for c in split_pdf]
                embs = emb_model.encode(texts, convert_to_numpy=True).tolist()
                ids = [f"pdf:{uploaded_pdf.name}:{i}" for i in range(len(split_pdf))]
                metas = [{"source": c.source, "section": c.section, "page": c.page} for c in split_pdf]
                collection.upsert(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
            st.success(f"Added {len(split_pdf)} chunks from {uploaded_pdf.name}!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "**Safety & Grounding Notice:**\n"
        "- Document answers are strictly grounded in verified campus files.\n"
        "- Fallback triggers whenever information is absent.\n"
        "- Calculations do not produce external side-effects."
    )

# Main Application Area
st.markdown('<div class="main-header">🎓 AI Campus Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Retrieve course information, check academic policies, calculate deadlines, and organize study schedules.</div>',
    unsafe_allow_html=True,
)

# Suggested Questions Feature (Shown when chat is empty)
if not st.session_state.messages:
    st.markdown("#### 💡 Suggested Questions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📖 What topics are covered in the course?", use_container_width=True):
            st.session_state.pending_prompt = "What topics are covered in the course?"
            st.rerun()

        if st.button("⚖️ What are the academic policies?", use_container_width=True):
            st.session_state.pending_prompt = "What are the academic policies?"
            st.rerun()

    with col2:
        if st.button("⏳ How many days remain until my deadline?", use_container_width=True):
            st.session_state.pending_prompt = "How many days remain until the final capstone deadline on May 10, 2027?"
            st.rerun()

        if st.button("📅 Help me create a study schedule.", use_container_width=True):
            st.session_state.pending_prompt = "Create a study schedule for Machine Learning, RAG, and LangGraph with 6 hours available."
            st.rerun()

    st.markdown("---")

# Render Conversation History
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    route = msg.get("route")
    path = msg.get("execution_path", [])
    sources = msg.get("sources", [])
    agent_steps = msg.get("agent_steps", [])

    with st.chat_message(role, avatar="👤" if role == "user" else "🎓"):
        # Display Route and Execution Path for Assistant responses
        if role == "assistant" and route:
            badge_class = f"route-{route.lower()}"
            st.markdown(f'<span class="route-badge {badge_class}">Route: {route}</span>', unsafe_allow_html=True)

            if path:
                path_str = " ➔ ".join(path)
                st.markdown(f'<div class="path-text">Execution Path: {path_str}</div>', unsafe_allow_html=True)

        st.markdown(content)

        # Expandable Sources Section
        if sources:
            with st.expander("📚 Retrieved Sources"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        # Expandable Agent Steps Section
        if agent_steps:
            with st.expander("🛠️ Agent Steps"):
                for step_idx, step in enumerate(agent_steps, start=1):
                    tool_name = step.get("tool", "unknown")
                    tool_input = step.get("input", {})
                    tool_output = step.get("output", "")
                    st.markdown(f"**Step {step_idx}: Tool `{tool_name}`**")
                    st.json({"input": tool_input, "output": tool_output})


# Handle Chat Input or Suggested Question Click
chat_query = st.chat_input("Ask a question about your courses, policies, deadlines, or study schedule...")
active_prompt = None

if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif chat_query:
    active_prompt = chat_query

if active_prompt:
    if not st.session_state.api_key:
        st.error("Please enter your Google Gemini API key in the sidebar before asking questions.")
    else:
        # 1. Append User Message
        st.session_state.messages.append({"role": "user", "content": active_prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # 2. Execute LangGraph Workflow
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Processing inquiry through LangGraph workflow..."):
                # Pass history excluding current user prompt
                history_for_graph = st.session_state.messages[:-1]
                result_state = execute_workflow(
                    question=active_prompt,
                    chat_history=history_for_graph,
                    api_key=st.session_state.api_key,
                )

                ans_route = result_state.get("route", "DOCS")
                ans_path = result_state.get("execution_path", [])
                ans_text = result_state.get("response", "")
                ans_sources = result_state.get("sources", [])
                ans_steps = result_state.get("agent_steps", [])

                # Render Route Badge & Path
                badge_class = f"route-{ans_route.lower()}"
                st.markdown(f'<span class="route-badge {badge_class}">Route: {ans_route}</span>', unsafe_allow_html=True)
                if ans_path:
                    path_str = " ➔ ".join(ans_path)
                    st.markdown(f'<div class="path-text">Execution Path: {path_str}</div>', unsafe_allow_html=True)

                st.markdown(ans_text)

                if ans_sources:
                    with st.expander("📚 Retrieved Sources"):
                        for src in ans_sources:
                            st.markdown(f"- `{src}`")

                if ans_steps:
                    with st.expander("🛠️ Agent Steps"):
                        for step_idx, step in enumerate(ans_steps, start=1):
                            tool_name = step.get("tool", "unknown")
                            tool_input = step.get("input", {})
                            tool_output = step.get("output", "")
                            st.markdown(f"**Step {step_idx}: Tool `{tool_name}`**")
                            st.json({"input": tool_input, "output": tool_output})

        # Save Assistant response to conversation history
        st.session_state.messages.append({
            "role": "assistant",
            "content": ans_text,
            "route": ans_route,
            "execution_path": ans_path,
            "sources": ans_sources,
            "agent_steps": ans_steps,
        })
