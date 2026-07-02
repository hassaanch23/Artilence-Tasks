"""Streamlit chat UI for the AI Research Assistant (agent + tools + memory)."""

import streamlit as st

from src import config
from src.research_assistant import ResearchAssistant

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎")
st.title("🔎 AI Research Assistant")
st.caption("LangChain agent · web search + calculator · buffer / long-term / vector memory")

# Fail early with a clear message if the key is missing.
if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(
        "Ask a research question. The agent decides when to **search the web** or "
        "**calculate**, remembers the conversation (buffer memory), and stores every "
        "exchange in **SQLite** + **vector memory** for context-aware answers that "
        "persist across sessions."
    )
    st.write(
        f"**Search backend:** "
        f"{'Google API' if config.google_search_enabled() else 'DuckDuckGo (keyless)'}"
    )
    if st.button("🗑️ New conversation"):
        for k in ("assistant", "messages"):
            st.session_state.pop(k, None)
        st.rerun()

# One assistant per browser session, so buffer memory survives Streamlit reruns.
if "assistant" not in st.session_state:
    st.session_state.assistant = ResearchAssistant(thread_id="streamlit")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

# Handle a new question.
if question := st.chat_input("Ask a research question..."):
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            answer = st.session_state.assistant.ask(question)
        st.write(answer)
    st.session_state.messages.append(("assistant", answer))
