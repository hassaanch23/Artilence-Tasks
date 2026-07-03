# Artilence Tasks

A phase-by-phase journey building **LangChain + LLM applications** — from prompt basics to
deployed, real-world AI products. **Each phase lives on its own branch**; `main` is this index.

| Phase | Branch | What it builds |
|-------|--------|----------------|
| 1 | [phase-1](https://github.com/hassaanch23/Artilence-Tasks/tree/phase-1) | **Foundations of LLMs + Prompt Engineering Playground** — a swappable OpenAI/Hugging Face client, zero-/few-shot/chain-of-thought prompt comparison, and a memory chatbot |
| 2 | [phase-2](https://github.com/hassaanch23/Artilence-Tasks/tree/phase-2) | **Introduction to LangChain** — an AI-Powered Q&A System with LCEL chains, buffer memory, and a Streamlit UI |
| 3 | [phase-3](https://github.com/hassaanch23/Artilence-Tasks/tree/phase-3) | **Agents & Tools** — an AI Research Assistant that web-searches and does math, with buffer + SQLite + vector memory |
| 4 | [phase-4](https://github.com/hassaanch23/Artilence-Tasks/tree/phase-4) | **Vector Databases & RAG** — an AI-Powered PDF Search Assistant (FAISS) that answers only from your document, with page citations |
| 5 | [phase-5](https://github.com/hassaanch23/Artilence-Tasks/tree/phase-5) | **Building AI Apps (capstone)** — an AI Customer Support Chatbot: FAQ RAG + live weather/stock APIs + per-session memory, deployed as **Streamlit *and* FastAPI** |
| 6 | [phase-6](https://github.com/hassaanch23/Artilence-Tasks/tree/phase-6) | **Mastery & Real-World Apps** — an AI Code Review Bot using structured output: reviews a file or a `git diff` into typed, severity-ranked findings (with a CI exit-code gate) |

## Tech

LangChain 1.x · OpenAI · FAISS · Streamlit · FastAPI · Python 3.12

## How it's organized

`main` stays as this index. To explore or run a phase, **check out its branch** — each one
has its own README with setup, concept notes, and a "why this approach" section written for
review. Every phase builds on the previous: prompts → chains → agents → RAG → deployment → mastery.
