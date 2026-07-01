# Phase 1 — Foundations of LLMs + Prompt Engineering Playground

A learning project with two goals:
1. **Understand** how Large Language Models work (concept notes below).
2. **Build** a Prompt Engineering Playground and a simple chatbot that run on
   either **OpenAI** or a local **Hugging Face** model through one swappable client.

---

## Project structure

```
Phase 1/
├── .env.example        # template for secrets/settings (copy to .env)
├── .gitignore          # keeps .env, venv, caches out of git
├── requirements.txt    # pinned dependencies
├── README.md           # you are here (concept notes + how to run)
├── src/
│   ├── config.py       # reads settings from .env
│   ├── llm_client.py   # the swappable OpenAI / Hugging Face wrapper
│   └── chatbot.py      # simple chatbot (added in a later step)
└── notebooks/
    └── playground.ipynb  # tokenization + prompt comparisons (later step)
```

---

## Setup

```bash
# 1. Create an isolated environment (so this project can't break others)
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local settings file from the template
cp .env.example .env
# (No API key? Leave it as-is — it defaults to the local Hugging Face model.)
```

---

## Concept notes

### 1. What is an LLM?
A **Large Language Model** is a neural network trained on huge amounts of text.
Its one job: **predict the next piece of text**. "Large" means billions of
*parameters* (the numbers it learned during training). That single skill —
next-token prediction — scales up into answering questions, writing code,
summarizing, and translating.

> **Mental model:** an LLM is a very sophisticated autocomplete. It does not
> "look things up" — it generates the most likely continuation from patterns it
> saw in training. This is also why it can be confidently wrong ("hallucinate").

### 2. How LLMs work (next-token prediction)
Text in → **tokens** → each token becomes a vector (**embedding**) → many
**transformer** layers → a probability for every possible next token → pick one
→ append it → repeat.

- **Temperature** controls randomness: low = focused/repeatable, high = creative/varied.
- **Context window** = the max tokens the model can consider at once (prompt + answer). Go over it and older text is dropped.

### 3. Tokenization & word embeddings
- **Tokenization** = splitting text into tokens, often *sub-words* not whole
  words. Earlier we saw `"Hello, LLM!"` → `[9906, 11, 445, 11237, 0]` (5 tokens).
  The model only ever sees these **numbers**, never letters. *Why it matters:*
  cost and context limits are counted in tokens.
- **Embeddings** = each token maps to a vector of numbers capturing *meaning*.
  Similar meanings sit close together, which is why the famous analogy
  `king - man + woman ≈ queen` works. Embeddings turn discrete tokens into math
  the model can reason over.

### 4. Attention & Transformers (brief)
- The **Transformer** (2017 paper *"Attention Is All You Need"*) is the
  architecture behind every modern LLM.
- **Attention** lets the model weigh *which other tokens matter* when
  interpreting a token. In *"The animal didn't cross the street because **it**
  was tired,"* attention links "it" → "animal".
- Stacking self-attention across many layers and "heads" is what captures
  context and long-range meaning. Everything else (more data, more parameters)
  just scales this up.

### 5. Fine-tuning vs. Prompt Engineering
| | Prompt Engineering | Fine-tuning |
|---|---|---|
| You change... | the **input** (the prompt) | the model's **weights** (re-train on your data) |
| Cost / speed | free-ish, instant, reversible | compute + data + time, slow to iterate |
| Use when... | almost always try this **first** | prompting can't get consistent format/style/jargon at scale |

Our whole playground is prompt engineering. Fine-tuning is out of scope for Phase 1.

### 6. The model landscape
- **Closed / API-only** (you call their servers, pay per token): OpenAI **GPT**
  (GPT-4 family, GPT-4o/`gpt-4o-mini`), Anthropic **Claude**, Google **Gemini**.
  Best quality, easiest to use — but data leaves your machine and it costs money.
- **Open-weights** (download + run yourself): Meta **LLaMA**, **Falcon**,
  **Mistral**, and thousands more. Free, private, but *you* provide the compute
  and quality scales with model size.
- **Hugging Face** = the "GitHub of models": a hub of open models + the
  `transformers` library to run them. This is our **free, no-key** path.

> *Note:* specific version numbers move fast (GPT-4-turbo, Claude's Opus/Sonnet/Haiku
> tiers, Gemini versions). Learn the **families and trade-offs**, not the version strings.

### 7. Prompting techniques (what Step 4 compares)
- **Zero-shot** — just ask, no examples: *"Classify the sentiment: 'I love this.'"*
- **Few-shot** — show a few examples first, then the real question. Teaches the
  format/behavior by demonstration.
- **Chain-of-Thought (CoT)** — tell the model to *"think step by step"* so it
  reasons before answering. Big accuracy boost on math/logic problems.

---

## Why the code is structured this way  *(talking points for PR review)*

- **`.env` for secrets** — API keys never touch git.
- **`config.py`** — one place reads settings; the rest of the code stays clean.
- **`llm_client.py` interface** — swap providers by adding one class, not editing the whole app.
- **Lazy imports** — you only install the library for the provider you actually use.
- **Notebook for exploring, `.py` for reusable logic** — clean diffs, testable code.

---

## How to run each piece

```bash
source .venv/bin/activate                 # activate the environment first

python -m src.hello                        # Step 3: one model call (sanity check)
python -m src.playground                   # Step 4: compare the 3 prompt styles
python -m src.chatbot --demo               # Step 5: scripted chatbot (proves memory)
python -m src.chatbot                      # Step 5: interactive chat ('exit' to quit)
jupyter lab notebooks/playground.ipynb     # Step 2: tokenization notebook
```

---

## Results & Findings
*(model: `google/flan-t5-base`, ~250M params, run locally on the Mac GPU via MPS)*

### Prompt-style comparison — `python -m src.playground`

| Task | zero-shot | few-shot | chain-of-thought |
|---|---|---|---|
| Sentiment *(easy)* | `negative` | `negative` | `negative` |
| Initials of "Grace Hopper" | `GH` | **`G.H.`** ✅ | rambling, wrong |
| Math: 3 boxes × 4 apples *(=12)* | `3` | `3` | shows work `…=36` (reasons, still wrong) |

**What we learned:**
- **Zero-shot** is enough for easy, well-known tasks (sentiment).
- **Few-shot** teaches an exact output *format* the instruction alone can't convey (`GH` → `G.H.`).
- **Chain-of-thought** makes the model *show reasoning* — but on a tiny model that reasoning can still be wrong (it got 36, not 12), and CoT can even *hurt* trivial tasks (it babbled on initials).
- **Model size dominates.** Every wrong answer above would very likely be correct on GPT-4/Claude.

### Chatbot — `python -m src.chatbot --demo`

```
You: Hi, my name is Sam.             Bot: Hi, I am Sam.
You: What is the capital of France?  Bot: France        (wrong — small-model limit)
You: What is my name?                Bot: Sam           (correct -> MEMORY WORKS)
```

The final "Sam" is the key result: only possible because past turns are fed back into the prompt.

### The headline takeaway
Prompt engineering changes model **behavior** (format, reasoning, memory) but cannot exceed the
model's underlying capability. The biggest quality lever is the model itself — which is exactly
why the swappable `llm_client.py` matters: set `LLM_PROVIDER=openai` in `.env` and every result
above improves, **with no code change**.

---

## Reproducing this environment

`requirements.txt` is the human-readable list of *what we need and why*.
`requirements-lock.txt` is the exact `pip freeze` snapshot for byte-for-byte reproducibility:

```bash
pip install -r requirements-lock.txt       # exact versions that produced the results above
```
