# Phase 3 — Advanced LangChain: Agents, Tools & Memory

An **AI Research Assistant** built with a **LangChain agent** that decides when to
**search the web** or **do math**, and remembers the conversation across three
layers of memory (**buffer**, **long-term SQLite**, and **vector**).

> Part of [Artilence Tasks](https://github.com/hassaanch23/Artilence-Tasks).
> Phase 1 (LLM foundations) and Phase 2 (LangChain chains + memory) live on the
> `phase-1` / `phase-2` branches. This phase builds directly on them.

---

## Project structure

```
├── .env.example              # settings template (copy to .env)
├── .gitignore                # keeps .env, venv, *.db out of git
├── requirements.txt          # pinned dependencies
├── README.md                 # you are here
├── src/
│   ├── config.py             # reads OpenAI / Google / DB settings from .env
│   ├── llm.py                # builds the ChatOpenAI model
│   ├── tools.py              # web_search (Google or DuckDuckGo) + calculator
│   ├── agent.py              # the agent (create_agent + tools + buffer memory)
│   ├── store.py              # long-term memory: SQLite persistence
│   ├── memory.py             # vector memory: semantic recall of past turns
│   └── research_assistant.py # ties agent + all three memories together
└── app.py                    # Streamlit chat UI
```

---

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenAI key; Google keys are optional
```

**Keys:** only `OPENAI_API_KEY` is required (chat **and** the embeddings used by
vector memory). Web search works **with no key** via DuckDuckGo; add Google
Custom Search keys later to switch to the official Google Search API.

---

## Run

```bash
streamlit run app.py            # the Research Assistant web app
```

### Try the pieces individually

```bash
python -m src.tools               # search + calculator tools on their own
python -m src.store               # SQLite save/read roundtrip (no key needed)
python -m src.memory              # vector recall by meaning
python -m src.agent               # agent: tool use + buffer memory
python -m src.research_assistant  # the full project: agent + all memories
```

---

## Concept notes

### 1. What is an Agent? (chain vs agent)
A **chain** (Phase 2) runs a **fixed** sequence: `prompt → model → parser`, every
time. An **agent** adds a decision: the model is given a set of **tools** and
decides, at runtime, **which tool to call, with what input, and when to stop**.

> **Mental model:** a chain is a railway (one track); an agent is a driver with a
> map and a toolbox — it picks the route based on the question.

The loop an agent runs is **ReAct** (*Reason + Act*):
1. **Reason** about the question.
2. **Act** — call a tool (search, calculator, …) if needed.
3. **Observe** the tool's result.
4. Repeat until it can answer.

Our assistant does exactly this: a math question → it calls `calculator`; a
"what's the latest…" question → it calls `web_search`; a "what's my name?"
question → no tool, just memory.

### 2. Agent types — and the LangChain 1.x reality ⚠️
Classic LangChain tutorials name agent **types** you set via
`initialize_agent(..., AgentType.X)`:

| Classic type | What it does |
|---|---|
| **Zero-shot ReAct** | Picks tools purely from their descriptions, no examples, no memory. |
| **Self-ask with search** | Splits a question into sub-questions and searches for each (multi-hop). |
| **Conversational ReAct** | ReAct + memory, for chatbots. |

**The catch:** `initialize_agent` and `AgentType` were **removed in LangChain
1.0** (this repo runs LangChain 1.3 — verified by importing them: they no longer
exist). Modern chat models have **native tool-calling**, so all those "types"
collapse into **one** tool-calling agent built with **`create_agent`**. The
behavior you used to select by *type* you now select by *prompt + tools +
whether you attach memory*. We explain the historical types (they're good mental
models) but implement the current API.

### 3. Tools — how the model "does" things
A **tool** is just a function the model may call. The `@tool` decorator turns a
Python function into one, and — crucially — the function's **name, type hints,
and docstring become the schema the model reads** to decide when/how to call it.
So a tool's docstring is really *prompt text aimed at the model*.

We ship two tools (`src/tools.py`):
- **`web_search`** — real-time info. Uses the **Google Custom Search API** if you
  set Google keys, otherwise **keyless DuckDuckGo**.
- **`calculator`** — exact arithmetic via a **safe** expression evaluator.

The brief also lists "Python execution" as a tool. LangChain's `PythonREPLTool`
runs **arbitrary** code — a real security risk — so we deliberately implement a
**whitelisted arithmetic evaluator** instead (parses the expression to an AST and
allows only numbers and `+ - * / // % **`). It refuses anything else:
`__import__('os')` → *"only numbers and … are allowed."*

### 4. Integrating external tools (Search, APIs, Databases)
- **Google Search API** — `web_search` calls Google's *Custom Search JSON API*
  directly over HTTPS (stdlib `urllib`). Set `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`
  and it switches automatically; otherwise DuckDuckGo keeps it running key-free.
- **Calling APIs for real-time data** — the search tool *is* a live API call; any
  REST endpoint can be wrapped as a `@tool` the same way.
- **Databases (SQL)** — `src/store.py` uses **SQLite** for long-term memory
  (every Q&A row). Queries are **parameterized** (`?` placeholders) so there's no
  SQL-injection surface. PostgreSQL is a one-function swap (see the code note).

### 5. Memory — the three types the brief asks for
LLMs are **stateless**; memory is what makes a conversation coherent.

| Type | What it is | Where in this repo |
|---|---|---|
| **Buffer memory** | Keep the recent message history and feed it back each turn. | `agent.py` — LangGraph **checkpointer** keyed by `thread_id`. |
| **Entity memory** | Track facts about *specific entities* (people/places) over time. | Explained here; its old `ConversationEntityMemory` class is deprecated in 1.x, so we cover its *job* via vector + SQLite recall. |
| **Vector memory** | Embed every exchange; recall the *semantically relevant* ones, even far back. | `memory.py` — `OpenAIEmbeddings` + `InMemoryVectorStore`. |

**Handling long-term memory:** buffer memory lives in RAM and dies with the
process. `store.py` persists every turn to **SQLite**, and on startup the
assistant **re-loads** that history into vector memory — so it can stay
context-aware **across restarts**, not just within one session.

How they combine in `research_assistant.py` for each question:
1. **Recall** semantically relevant past exchanges (vector) → inject as context.
2. **Run** the agent (buffer memory auto-applied via `thread_id`).
3. **Persist** the new turn to SQLite **and** vector memory.

---

## Why the code is structured this way  *(talking points for PR review)*

| Decision | We chose | Instead of | Why |
|---|---|---|---|
| Agent constructor | **`create_agent`** (LangChain 1.x) | `initialize_agent` + `AgentType` | The classic API is **removed** in LangChain 1.0; `create_agent` is the supported, LangGraph-based replacement. |
| Reasoning loop | **Native tool-calling** (via `create_agent`) | Hand-parsing `Thought:/Action:` text | Structured tool calls are far more reliable than regex-parsing model prose. |
| Web search | **Direct Google CSE API + DuckDuckGo fallback** | `langchain-google-community` wrapper | Fewer/lighter deps, transparent code, and it **runs keyless** out of the box. |
| "Python" tool | **Whitelisted `ast` calculator** | `PythonREPLTool` (arbitrary exec) | Arbitrary code execution is a security risk; an AST whitelist is safe by construction. |
| Long-term store | **SQLite** (stdlib) | PostgreSQL | Zero-config single file; `store.py` isolates the connection so Postgres is a one-spot swap. |
| Vector memory | **`InMemoryVectorStore` + OpenAI embeddings** | FAISS / Chroma | No extra infrastructure; enough to *demonstrate* semantic recall. |
| Buffer memory | **LangGraph checkpointer** (`thread_id`) | Manual `history=[]` list (Phase 1/2) | Built into the agent runtime; correct across multiple `.invoke()` calls and threads. |
| Secrets / config | **`.env` + `config.py`** | Hard-coding keys | Same discipline as Phases 1–2: keys never touch git. |

---

## How this builds on Phases 1 & 2

| Phase 1 (hand-built) | Phase 2 (LangChain chains) | Phase 3 (agents) |
|---|---|---|
| Manual `llm_client` | `ChatOpenAI` | `ChatOpenAI` inside an **agent** |
| Fixed prompt→model | **Chain** (`prompt \| model \| parser`) | **Agent** that *chooses* tools at runtime |
| `history = []` list | `MessagesPlaceholder` buffer | **Checkpointer** + **SQLite** + **vector** memory |
| — | — | **Tools**: web search, calculator |

The theme across all three: *the model is only as useful as what you connect it
to.* Phase 3 connects it to **actions** (tools) and **durable context** (memory).

---

## Results & Findings

### Agent uses tools + remembers — `python -m src.agent`
```
You: What is 128 * 17?
Bot: 128 multiplied by 17 equals 2176.               <- called the calculator tool
You: And what did I just ask you to compute?
Bot: You asked me to compute the product of 128 and 17.   <- BUFFER MEMORY works
```

### The full project — `python -m src.research_assistant`
```
You: My name is Hassaan and I'm researching solar energy.
Bot: Hi Hassaan! ... What specific aspects of solar energy are you researching?
You: Search the web for a recent advance in solar panel efficiency.
Bot: Here are some recent advances ... 3. Bifacial Solar Panels ... (with source URLs)   <- web_search tool
You: What is my name and what am I researching?
Bot: Your name is Hassaan, and you are researching solar energy.        <- MEMORY works
```
All three turns are then persisted to SQLite (`3 Q&A pairs`), so a *restarted*
session can still recall them via vector memory.

### Security guard — the calculator refuses code
```
calculator("2 ** 10 + 24")     -> 1048
calculator("__import__('os')") -> "calculator error: only numbers and + - * / // % ** are allowed."
```

### Headline takeaway
A **chain** answers; an **agent** *acts* — it reaches for the right tool and
remembers what happened. The two levers that make it work are the **tool
descriptions** (they are the model's instructions) and the **memory layers**
(buffer for now, SQLite + vectors for later).

---

## Reproducing this environment

Tested with **Python 3.12.2** and:

```
langchain==1.3.11   langchain-openai==1.3.3   langgraph==1.2.7
streamlit==1.58.0   ddgs==9.14.4              python-dotenv==1.2.2
```

`requirements.txt` lists the floors with a note on why each is needed; run
`pip freeze > requirements-lock.txt` in a clean venv for a byte-for-byte lock.
