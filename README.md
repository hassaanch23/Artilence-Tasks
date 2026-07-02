# Phase 5 — Building AI-Powered Applications (Capstone)

An **AI Customer Support Chatbot** for a fictional e-bike company (*Aurora Mobility*)
that ties together everything from Phases 1–4: it **answers FAQs from a vector
database (RAG)**, **fetches real-time data from live APIs** (weather + stock),
**remembers the conversation**, and ships as **two deployment targets** — a
**Streamlit** app and a **FastAPI** service.

> Part of [Artilence Tasks](https://github.com/hassaanch23/Artilence-Tasks).
> This is the capstone; Phases 1–4 live on their own branches.

---

## Project structure

```
├── .env.example / .gitignore / requirements.txt / README.md
├── data/
│   └── faqs.md              # the FAQ knowledge base (what the bot answers from)
├── src/
│   ├── config.py            # settings from .env
│   ├── llm.py  / embeddings.py
│   ├── knowledge.py         # FAQs → per-Q&A chunks → FAISS   (Phase 4 RAG)
│   ├── tools.py             # search_faqs (RAG) + get_weather + get_stock (live APIs)
│   ├── agent.py             # support agent: create_agent + tools + memory (Phase 3)
│   ├── support.py           # SupportChatbot — memory keyed by session_id
│   └── multi_agent.py       # supervisor → specialist agents (multi-agent demo)
├── app.py                   # deployment #1: Streamlit chat UI
└── api.py                   # deployment #2: FastAPI /chat + /health
```

---

## Architecture

```
                        ┌──────────────── SupportChatbot (support.py) ───────────────┐
 user message  ──▶  app.py (Streamlit)   │   support agent  (create_agent, Phase 3)   │
     or        ──▶  api.py  (FastAPI)  ──▶│     • buffer memory, keyed by session_id   │
 HTTP /chat                              │     • picks a tool per turn:                │
                                         │         ├─ search_faqs → FAISS (Phase 4 RAG)│
                                         │         ├─ get_weather → Open-Meteo (live)   │
                                         │         └─ get_stock   → Yahoo (live)        │
                                         └────────────────────────────────────────────┘
```

One agent, three tools, two front doors. Each requirement of the brief maps to one
piece: **FAQ = `search_faqs` (RAG)**, **real-time data = `get_weather`/`get_stock`**,
**context = the checkpointer memory**, **deployment = `app.py` + `api.py`**.

---

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenAI key
```

Only `OPENAI_API_KEY` is required. The live tools use **keyless** public APIs
(Open-Meteo for weather, Yahoo Finance for stock), so nothing else is needed.

---

## Run

```bash
# Deployment #1 — Streamlit chat UI
streamlit run app.py

# Deployment #2 — FastAPI service (interactive docs at /docs)
uvicorn api:app --reload
curl -X POST http://127.0.0.1:8000/chat \
     -H 'content-type: application/json' \
     -d '{"message": "How long is the warranty?", "session_id": "u1"}'
```

### Try the pieces individually
```bash
python -m src.knowledge      # build the FAQ index + a sample retrieval
python -m src.tools          # exercise all three tools directly
python -m src.support        # agent + memory (proves session isolation)
python -m src.multi_agent    # supervisor routing to specialist agents
```

---

## Concept notes

### 1. Multi-agent systems
A **multi-agent system** uses several LLM "agents," each specialised for a role, and
a **supervisor** that routes work between them (the idea behind AutoGPT-style
autonomous assistants).

This project ships **both** shapes so you can compare:
- **One agent, many tools** (`agent.py`) — the model picks a tool (`search_faqs`,
  `get_weather`, `get_stock`) per turn. This *is* task-specific routing, and it's the
  main app: simpler, cheaper, fewer failure modes.
- **Supervisor + specialists** (`multi_agent.py`) — a router LLM sends the question to
  a *product* specialist (FAQ tools) or a *data* specialist (live-API tools), each its
  own agent with its own prompt and tool subset.

**When to use which:** reach for multiple agents when roles need genuinely different
prompts, tools, or guardrails (e.g. a "billing" agent that can issue refunds vs. a
read-only "info" agent). Otherwise one tool-using agent is usually enough — don't add
agents you don't need.

### 2. Deploying LangChain apps
| Option | Good for | We use it |
|---|---|---|
| **Streamlit** | Fast interactive demo UIs | `app.py` — the chat interface |
| **FastAPI** | Production HTTP APIs a website/app/service calls | `api.py` — `POST /chat`, `GET /health` |
| **Gradio** | Quick ML demos, shareable links | (alternative to Streamlit) |

**Cloud:** the FastAPI app is a standard ASGI service — deploy it to **AWS**
(Lambda + API Gateway via Mangum, or ECS), **Azure**, **Vercel**, or any container
host; Streamlit deploys to Streamlit Community Cloud or a container. **Low-latency
tips:** build the FAISS index once at startup (we do — `get_or_build_index`), reuse a
single chatbot instance (we do), stream tokens for perceived speed, and cache
embeddings so a document is embedded only once.

### 3. LLM evaluation & fine-tuning *(conceptual — out of build scope)*
- **RAG / embeddings vs. fine-tuning.** This whole app answers company questions with
  **RAG**, *not* fine-tuning. RAG injects knowledge at *query* time (cheap, instantly
  updatable — edit `faqs.md` and re-index), while fine-tuning bakes it into the
  *weights* (expensive, needs GPUs + labelled data, and is stale the moment facts
  change). **Rule of thumb:** use RAG to give a model *knowledge*, fine-tune to change
  its *behaviour/format/tone*.
- **Local models.** For privacy or cost you can run open-weight models (**LLaMA**,
  **Mistral**, **Falcon**) locally (e.g. via Ollama) and point `llm.py`/`embeddings.py`
  at them — the same swap-point pattern from Phase 1.
- **Evaluation.** For a support bot you'd measure retrieval hit-rate (is the right FAQ
  retrieved?), answer groundedness (is it supported by the context?), and refusal
  correctness (does it decline when the FAQ doesn't cover it?).

---

## Why the code is structured this way  *(talking points for PR review)*

| Decision | We chose | Instead of | Why |
|---|---|---|---|
| Core architecture | **One agent + 3 tools** | A full multi-agent swarm | Tool-selection is routing; simplest thing that meets the brief. `multi_agent.py` still demos the supervisor pattern. |
| Knowledge base | **FAISS** (local) | Pinecone (cloud) | Runs keyless/offline; `knowledge.py` isolates it so Pinecone is a one-file swap. |
| FAQ chunking | **One chunk per `## ` Q&A** | Size-based splitting | Size-splitting merged ~5 Q&As per chunk and blurred retrieval; per-entry chunks give rank-1 hits. |
| Live data | **Keyless Open-Meteo + Yahoo Finance** | Keyed APIs (OpenWeather, Alpha Vantage) | Runs out of the box; also *fixes staleness* — a live API returns today's data, unlike the model's training cutoff. |
| Memory | **Checkpointer keyed by `session_id`** | A global history | Isolates each user/session — required for a deployed multi-user service. |
| Deployment | **Streamlit *and* FastAPI** | Just one | Covers the brief and shows the real-product path (an HTTP API), not only a demo UI. |
| Fine-tuning | **RAG instead** | Actually fine-tuning | Instant, cheap, updatable knowledge; fine-tuning is for behaviour, not facts. |

---

## How this is the capstone of Phases 1–5

| Phase | What it contributed here |
|---|---|
| **1** | Provider swap-points (`llm.py`, `embeddings.py`); embeddings concept. |
| **2** | LCEL + the idea of composable chains. |
| **3** | The **agent** (`create_agent`), **tools**, and **buffer memory** (checkpointer). |
| **4** | **RAG** over a vector DB — reused verbatim for FAQ search. |
| **5** | **Multi-agent** routing + **deployment** (Streamlit **and** FastAPI). |

The support bot is literally *Phase 3's agent* holding *Phase 4's RAG* as one of its
tools, remembering context per user, exposed through two deployment surfaces.

---

## Results & Findings

**FAQ retrieval (`python -m src.knowledge`)** — the FAQ splits into **14 one-per-Q&A
chunks**, and each test query retrieves the exact right entry at **rank 1** (including
the semantic match *"can I ride it in the rain?"* → *"Is the Aurora X1 water
resistant?"* → IPX5).

**The chatbot (`python -m src.support`)** — memory is isolated per `session_id`:
```
You (s1): Hi, what's your return policy?   Bot: ...within 30 days ... free US return shipping.
You (s1): How many days did you say?       Bot: You have 30 days ...          <- memory works
You (s2): What did I just ask you?          Bot: (no memory of s1)             <- sessions isolated
```

**Live tools + agent routing (`python -m src.multi_agent`)**:
```
"warranty on the Aurora X1?"   -> [product specialist] two-year frame / one-year battery
"weather in Tokyo right now?"  -> [data specialist]    21.3°C, humidity 89%, wind 1.3 km/h
```

**FastAPI deployment (`uvicorn api:app`)**:
```
GET  /health -> {"status":"ok","company":"Aurora Mobility"}
POST /chat   -> "The Aurora X1 is priced at $1,899 USD..."     (RAG answer over HTTP)
POST /chat   -> "...it is rated IPX5..."   (follow-up; same session_id remembers "it")
```

### Headline takeaway
Everything the four earlier phases taught — prompting, chains, agents+tools+memory,
and RAG — composes into one deployable product. The agent decides *when* to look up an
FAQ (RAG), *when* to call a live API (fresh data), and *what* the conversation is about
(memory); FastAPI turns it from a demo into a service other software can call.

---

## Reproducing this environment

Tested with **Python 3.12.2** and:

```
langchain==1.3.11     langchain-openai==1.3.3    langchain-community==0.4.2
langgraph==1.2.7      faiss-cpu==1.14.3          langchain-text-splitters==1.1.2
fastapi + uvicorn     streamlit==1.58.0          python-dotenv==1.2.2
```

Live tools call keyless public APIs (Open-Meteo, Yahoo Finance). Run
`pip freeze > requirements-lock.txt` in a clean venv for a byte-for-byte lock.
