"""Streamlit deployment of the AI Customer Support Chatbot."""

import streamlit as st

from src import config
from src.support import SupportChatbot

st.set_page_config(page_title=f"{config.COMPANY} Support", page_icon="💬")
st.title(f"💬 {config.COMPANY} — AI Support")
st.caption("FAQ (RAG) · live weather + stock tools · remembers the conversation")

# Fail early with a clear message if the key is missing.
if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

with st.sidebar:
    st.header("Try asking")
    st.markdown(
        "- *How long is the warranty?* (FAQ)\n"
        "- *What's your return policy?* (FAQ)\n"
        "- *What's the weather in London?* (live API)\n"
        "- *Price of AAPL?* (live API)\n\n"
        "Then a follow-up — it remembers the context."
    )
    if st.button("🗑️ New chat"):
        for k in ("bot", "messages"):
            st.session_state.pop(k, None)
        st.rerun()

# One chatbot per browser session; buffer memory survives Streamlit reruns.
if "bot" not in st.session_state:
    st.session_state.bot = SupportChatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

# Handle a new message.
if question := st.chat_input("How can we help?"):
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.bot.ask(question, session_id="streamlit")
        st.write(answer)
    st.session_state.messages.append(("assistant", answer))
