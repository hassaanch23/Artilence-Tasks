"""Streamlit UI for the AI Code Review Bot: paste code, get severity-ranked findings."""

from pathlib import Path

import streamlit as st

from src import config
from src.reviewer import review_code

_SAMPLE = Path("examples/sample_code.py")
_DEFAULT = _SAMPLE.read_text() if _SAMPLE.exists() else "def divide(a, b):\n    return a / b\n"
_COLOR = {"high": "🔴", "medium": "🟡", "low": "🔵"}

st.set_page_config(page_title="AI Code Review Bot", page_icon="🔍")
st.title("🔍 AI Code Review Bot")
st.caption("Upload or paste code → typed, severity-ranked findings (bugs · security · performance · style)")

if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

uploaded = st.file_uploader(
    "Upload a code file",
    type=["py", "js", "ts", "java", "go", "rb", "php", "c", "cpp", "cs",
          "rs", "kt", "swift", "sql", "sh", "txt"],
)
if uploaded is not None:
    # Uploaded file wins: read its bytes and review those directly.
    code = uploaded.getvalue().decode("utf-8", errors="replace")
    filename = uploaded.name
    st.caption(f"Loaded **{filename}** — {len(code.splitlines())} lines")
    with st.expander("Preview file"):
        st.code(code)
else:
    # No upload → fall back to pasting (prefilled with the flawed sample).
    filename = st.text_input("Filename (for context)", "sample_code.py")
    code = st.text_area("…or paste code here", value=_DEFAULT, height=320)

if st.button("🔍 Review") and code.strip():
    with st.spinner("Reviewing..."):
        result = review_code(code, filename)

    st.subheader("Summary")
    st.write(result.summary)

    if not result.findings:
        st.success("✅ No issues found.")
    else:
        counts = {s: sum(f.severity == s for f in result.findings) for s in ("high", "medium", "low")}
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High", counts["high"])
        c2.metric("🟡 Medium", counts["medium"])
        c3.metric("🔵 Low", counts["low"])
        for f in result.findings:
            with st.expander(f"{_COLOR[f.severity]} line {f.line} · {f.category} — {f.issue}"):
                st.markdown(f"**Severity:** {f.severity}")
                st.markdown(f"**Suggestion:** {f.suggestion}")
