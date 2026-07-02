# Phase 2 — Introduction to LangChain

An **AI-Powered Q&A System** built with **LangChain (LCEL)** + **OpenAI**, with
conversation **memory** and a **Streamlit** chat UI.

> Part of [Artilence Tasks](https://github.com/hassaanch23/Artilence-Tasks).
> Phase 1 (LLM foundations) lives on the `phase-1` branch.

## Project structure

```
├── .env.example        # settings template (copy to .env)
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── config.py       # reads OPENAI_API_KEY / model from .env
│   ├── llm.py          # builds the LangChain model (ChatOpenAI)
│   ├── chain.py        # prompt templates + the LCEL chain
│   └── qa.py           # Q&A engine with conversation (buffer) memory
└── app.py              # Streamlit chat UI
```

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your OpenAI key into .env
```

## Run

```bash
streamlit run app.py            # the Q&A web app
```

### Try the pieces individually

```bash
python -m src.llm      # one ChatOpenAI call
python -m src.chain    # prompt templates + the LCEL chain
python -m src.qa       # Q&A with memory (proves it remembers)
```

Memory demo (`python -m src.qa`):

```
You: My name is Hassaan.             Bot: Nice to meet you, Hassaan! ...
You: What is the capital of France?  Bot: The capital of France is Paris.
You: What is my name?                Bot: Your name is Hassaan.   <- memory works
```

### Design note — memory
LangChain 1.x **deprecated** `RunnableWithMessageHistory` (it points to LangGraph).
Rather than add a whole framework for a simple Q&A, `qa.py` implements **buffer
memory directly**: a running message list fed into the prompt's `MessagesPlaceholder`.
Warning-free, dependency-light, and it mirrors the manual `history = []` from Phase 1.

## Concept notes

### 1. What is LangChain? Why use it?
LangChain is a **framework for building apps powered by LLMs**. Instead of calling
a model's raw API and wiring everything yourself (Phase 1), it gives you standard,
composable building blocks.

Why it's useful:
- **Model-agnostic** — swap OpenAI ↔ Hugging Face ↔ others by changing one class.
- **Composable** — connect steps into *chains* (prompt → model → parse → …).
- **Batteries included** — memory, retrieval (RAG), agents, output parsing.
- **Ecosystem** — a huge library of ready-made integrations.

> **Honest trade-off (for the PR):** LangChain adds an abstraction layer. For a
> single raw call it's overkill — it pays off when you compose multi-step,
> memory-ful, multi-model apps.

### 2. Core components & architecture
LangChain 1.x is split into **modular packages** (same "install only what you use"
idea as Phase 1):
- `langchain-core` — base abstractions (Runnable, prompts, messages, parsers).
- `langchain-openai` / `langchain-huggingface` / … — provider integrations.
- `langchain` — higher-level helpers.

The building blocks:
- **Chat model** — a wrapper around an LLM (`ChatOpenAI`): messages in → reply out.
- **Prompts** — `PromptTemplate` / `ChatPromptTemplate`: reusable templates with `{variables}` filled at runtime.
- **Chains** — pipelines connecting steps, built with **LCEL** (the `|` pipe).
- **Memory** — remembers past turns so a conversation has context.
- **Output parsers** — shape the raw reply (`StrOutputParser` → plain string).

### 3. LCEL — the `|` pipe (modern way to build chains)
LCEL (LangChain Expression Language) connects components with `|`, like Unix pipes:

```python
chain = prompt | model | parser
answer = chain.invoke({"question": "..."})
```

Data flows left → right: your input fills the **prompt** → goes to the **model** →
the reply goes to the **parser**. Every piece is a **Runnable** (has `.invoke()`,
`.stream()`, `.batch()`), so they snap together uniformly.

> *Curriculum mapping:* the assignment says "Chains (sequential processing)". LCEL
> `a | b | c` **is** sequential processing. The older `LLMChain` class did the same
> job but is now deprecated in favor of LCEL.

### 4. Memory (buffer memory)
LLMs are **stateless** — they forget between calls (the exact problem we solved by
hand in Phase 1's `chatbot.py`). LangChain's **memory** stores the conversation and
feeds it back automatically.
- **Buffer memory** = keep the full conversation history and prepend it each turn.
- In LangChain 1.x we implement it with `RunnableWithMessageHistory` wrapping our
  chain + an `InMemoryChatMessageHistory` store. (The old `ConversationBufferMemory`
  did this; the message-history approach is the current equivalent.)

### 5. Installing (what we did)
```bash
pip install langchain langchain-openai streamlit python-dotenv
```
`langchain-openai` pulls in `langchain-core`. We reuse Phase 1's `.env` for the `OPENAI_API_KEY`.

## How this builds on Phase 1

| Phase 1 (hand-built) | Phase 2 (LangChain) |
|---|---|
| `llm_client.py` | `ChatOpenAI` |
| `prompts.py` | `PromptTemplate` |
| manual prompt→model | **Chains** (LCEL) |
| `chatbot.py` history list | **Memory** |
