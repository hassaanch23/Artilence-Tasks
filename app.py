"""Streamlit UI for the AI-Powered PDF Search Assistant (RAG over PDFs)."""

import os
import tempfile

import streamlit as st

from src import config
from src.loader import load_and_split
from src.rag import answer_with_sources
from src.vectorstore import build_index

st.set_page_config(page_title="PDF Search Assistant", page_icon="📄")
st.title("📄 AI-Powered PDF Search Assistant")
st.caption("RAG · LangChain + FAISS · answers grounded in your PDF, with page citations")

# Fail early with a clear message if the key is missing.
if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

with st.sidebar:
    st.header("1. Load a PDF")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None and st.button("📥 Index this PDF"):
        with st.spinner("Reading + embedding..."):
            # PyPDFLoader needs a file path, so write the upload to a temp file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                chunks = load_and_split(tmp_path)
                st.session_state.index = build_index(chunks)
                st.session_state.messages = []
            finally:
                os.unlink(tmp_path)  # never keep the user's file in the repo
        st.success(f"Indexed {len(chunks)} chunks. Ask away!")

# Nothing to search until a PDF is indexed.
if "index" not in st.session_state:
    st.info("Upload a PDF in the sidebar and click **Index this PDF** to begin.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

# Handle a new question.
if question := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, docs = answer_with_sources(st.session_state.index, question)
        st.write(answer)
        with st.expander("📚 Sources (retrieved passages)"):
            for d in docs:
                page = d.metadata.get("page")
                page_str = page + 1 if isinstance(page, int) else "?"
                st.markdown(f"**Page {page_str}** — {d.page_content[:300].strip()}…")
    st.session_state.messages.append(("assistant", answer))
