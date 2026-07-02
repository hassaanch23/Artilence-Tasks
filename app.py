"""Streamlit chat UI for the AI-Powered Q&A System (LangChain + OpenAI + memory)."""

import streamlit as st

from src import config
from src.qa import QASession

st.set_page_config(page_title="AI-Powered Q&A", page_icon="🤖")
st.title("🤖 AI-Powered Q&A System")
st.caption("LangChain (LCEL) + OpenAI · remembers the conversation")

# Fail early with a clear message if the key is missing.
if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

# One Q&A session per browser session, so memory survives Streamlit's reruns.
if "qa" not in st.session_state:
    st.session_state.qa = QASession()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About")
    st.write(
        "Ask a question, then a follow-up — it remembers the context "
        "(buffer memory)."
    )
    if st.button("🗑️ Clear conversation"):
        st.session_state.qa = QASession()
        st.session_state.messages = []
        st.rerun()

# Replay the conversation so far.
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

# Handle a new question.
if question := st.chat_input("Ask me anything..."):
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.qa.ask(question)
        st.write(answer)
    st.session_state.messages.append(("assistant", answer))
