# Phase 4 — Vector Databases & Retrieval-Augmented Generation (RAG)

An **AI-Powered PDF Search Assistant**: upload a PDF, and ask questions answered
**only** from its contents — with page citations. Built with **LangChain**,
**FAISS** (a local vector database), **OpenAI embeddings**, and **Streamlit**.

> Part of [Artilence Tasks](https://github.com/hassaanch23/Artilence-Tasks).
> Phases 1–3 live on their own branches. Phase 3 already used embeddings + a
> vector store for *memory*; Phase 4 makes vector search the **whole point**.

---

## Project structure

```
├── .env.example              # settings template (copy to .env)
├── .gitignore                # keeps .env, venv, the FAISS index out of git
├── requirements.txt          # pinned dependencies
├── README.md                 # you are here
├── data/
│   └── sample.pdf            # a small fictional doc, so it runs out of the box
├── src/
│   ├── config.py             # OpenAI / chunking / index settings from .env
│   ├── llm.py                # builds the ChatOpenAI model
│   ├── embeddings.py         # builds the embedding model (OpenAI ↔ HF swap point)
│   ├── loader.py             # PDF → pages (PyPDFLoader) → chunks (splitter)
│   ├── vectorstore.py        # build / save / load the FAISS index
│   ├── rag.py                # the RAG chain: retrieve → prompt → LLM → cited answer
│   └── ingest.py             # CLI: embed a PDF into the FAISS index
└── app.py                    # Streamlit: upload a PDF, ask, get answers + sources
```

---

## How RAG works (the two pipelines)

RAG has an **offline** step (do once per document) and an **online** step (every question):

```
INGEST (once)                          QUERY (every question)
──────────────                         ──────────────────────
 PDF                                    question
  │ PyPDFLoader                          │ embed (same model!)
  ▼                                      ▼
 pages                                  query vector
  │ RecursiveCharacterTextSplitter       │ FAISS similarity search (top-k)
  ▼                                      ▼
 chunks                                 most relevant chunks
  │ OpenAIEmbeddings                     │ stuff into the prompt as CONTEXT
  ▼                                      ▼
 vectors ──► FAISS index ──(save)──►    LLM answers FROM that context ──► cited answer
```

The key rule: **the same embedding model must embed both the documents and the
question**, or their vectors live in different spaces and "nearest" means nothing.

---

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenAI key
```

Only `OPENAI_API_KEY` is required (used for **both** the chat model and embeddings).

---

## Run

```bash
# 1. Ingest a PDF once -> builds & saves the FAISS index
python -m src.ingest data/sample.pdf

# 2a. Ask questions from the terminal
python -m src.rag

# 2b. …or use the web app (upload any PDF, then chat)
streamlit run app.py
```

### Try the pieces individually
```bash
python -m src.loader data/sample.pdf   # PDF -> chunks (see the counts)
python -m src.llm                      # one ChatOpenAI call
python -m src.ingest data/sample.pdf   # embed + save the index
python -m src.rag                      # retrieve + answer with citations
```

---

## Concept notes

### 1. What is a vector database? Why use one with LLMs?
An **embedding** turns text into a vector (a list of numbers) that captures its
**meaning**; similar meanings get nearby vectors. A **vector database** stores
many of these vectors and answers one question fast: *"which stored vectors are
closest to this query vector?"* (nearest-neighbour search).

Why with LLMs? An LLM has a **fixed context window** and **no knowledge of your
private documents**. You can't paste a 300-page PDF into every prompt. A vector
DB lets you fetch just the handful of passages relevant to each question and feed
*those* to the model. It is the LLM's long-term, searchable memory.

### 2. Popular vector databases

| DB | Runs where | Key trait |
|---|---|---|
| **FAISS** | **Local** (a library, saved to a file) | Free, offline, fast. **We use this.** |
| **Chroma** | Local (embedded) | Easy local persistence, metadata filtering. |
| **Pinecone** | Cloud (managed) | Scales to billions of vectors; needs an account + API key. |
| **Weaviate** | Cloud or self-hosted | Vector + keyword hybrid search, schema/graph features. |

We chose **FAISS** because it needs no server or key — it runs on your laptop and
saves to `faiss_index/`. Swapping to Pinecone touches only `vectorstore.py`.

### 3. What is RAG, and why it matters
**Retrieval-Augmented Generation** = *retrieve relevant text, then let the model
generate an answer grounded in it.* It fixes the three biggest LLM weaknesses:
- **Hallucination** — the model answers from real passages, not guesses.
- **Stale / private knowledge** — it can use documents the model never trained on.
- **No citations** — because we know which chunks we retrieved, we can cite them.

Our prompt enforces this: *"answer using ONLY the context; if it's not there, say
you don't know."* That is why the assistant refuses "What is the capital of
France?" when the PDF is about something else — grounding over guessing.

### 4. Embeddings — OpenAI or Hugging Face
`embeddings.py` is the single swap point. We default to OpenAI
`text-embedding-3-small` (fast, cheap, high quality, and we already use the key).
For a **fully local, no-key** setup, swap in a Hugging Face
`sentence-transformers` model — the comment in `embeddings.py` shows the one-line
change, and nothing else in the app needs to change.

### 5. Chunking — why we split the PDF
We split pages into ~1000-character chunks with 150 characters of overlap
(`RecursiveCharacterTextSplitter`). Too big and the one relevant sentence is
diluted among hundreds of others; too small and it loses context. Overlap stops a
sentence that straddles a boundary from being cut in half.

---

## Why the code is structured this way  *(talking points for PR review)*

| Decision | We chose | Instead of | Why |
|---|---|---|---|
| Vector DB | **FAISS** (local, saved to disk) | Pinecone / Weaviate (cloud) | No account/key; runs offline & free. Isolated in `vectorstore.py` for a one-file swap. |
| Persistence | **Save the index to disk** | Re-embed on every run | Embedding costs money + time; ingest once, query forever. |
| PDF loading | **`PyPDFLoader`** (LangChain) | Raw `pypdf` parsing | The brief asks for LangChain document loaders; page-number metadata comes free for citations. |
| Embeddings | **OpenAI**, swappable to **HF** | Hard-coding one provider | Covers the brief's "OpenAI *or* Hugging Face" with a single swap point. |
| RAG chain | **LCEL** (`retriever \| prompt \| llm \| parser`) | Deprecated `RetrievalQA` | Builds on the LCEL from Phase 2; transparent and current. |
| Grounding | **"Answer only from context; else say you don't know"** | Free-form answering | Turns the LLM from a guesser into a document-grounded search engine. |
| Citations | **Return retrieved docs + page tags** | Answer text only | Lets the UI show *where* each answer came from — trust + verifiability. |

> **Ecosystem note (worth saying in the PR):** `langchain-community` (which hosts
> `PyPDFLoader` + the FAISS wrapper) is being sunset in favour of standalone
> integration packages. It is still the correct home for these classes in this
> version; the code is isolated in `loader.py` / `vectorstore.py` so a future
> migration is contained.

---

## How this builds on Phases 1–3

| | Contribution that Phase 4 reuses |
|---|---|
| **Phase 1** | Embeddings concept (`king − man + woman ≈ queen`) — now used for real. |
| **Phase 2** | **LCEL** chains (`prompt \| model \| parser`) — the RAG chain is one. |
| **Phase 3** | Embeddings + a vector store as **memory** — Phase 4 makes retrieval the core. |

Phase 3 retrieved past *conversation turns*; Phase 4 retrieves *document chunks*.
Same machinery (embed → store → nearest-neighbour search), aimed at documents.

---

## Results & Findings

Ingesting the bundled `data/sample.pdf` (a fictional "Project Nimbus" doc) and
querying — `python -m src.ingest data/sample.pdf` then `python -m src.rag`:

```
Ingested data/sample.pdf: 2 chunks -> FAISS index at 'faiss_index/'

Q: What is Project Nimbus and when was it launched?
A: Project Nimbus is an internal weather-analytics platform built by
   Artilence, launched on March 14, 2023 (page 1).        <- GROUNDED + cited

Q: What is the capital of France?
A: I don't know.                                           <- refuses (not in the PDF)
```

The first answer is only possible via retrieval — "Project Nimbus" is invented,
so the fact **cannot** come from the model's training data. The second proves the
grounding guard: the model plainly *knows* the capital of France, but answers only
from the document.

### Headline takeaway
A raw LLM answers from memory (and sometimes makes things up). A **RAG** system
answers from **your documents** — retrieve the right passages, ground the model in
them, and cite the source. The vector database is what makes "find the right
passages" fast and scalable.

---

## Reproducing this environment

Tested with **Python 3.12.2** and:

```
langchain==1.3.11        langchain-openai==1.3.3      langchain-community==0.4.2
langchain-text-splitters==1.1.2   faiss-cpu==1.14.3   pypdf==6.14.2
streamlit==1.58.0        python-dotenv==1.2.2
```

`data/sample.pdf` was generated with `reportlab` (a dev-only fixture, not a
runtime dependency). Run `pip freeze > requirements-lock.txt` in a clean venv for
a byte-for-byte lock.
