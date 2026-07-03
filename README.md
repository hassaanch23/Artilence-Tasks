# Phase 6 — AI Code Review Bot (Mastery & Real-World Applications)

An **AI Code Review Bot**: give it a source file or a `git diff`, and it returns
**typed, severity-ranked findings** (bugs, security, performance, style) with line
numbers and concrete fixes — usable from the terminal, in CI, or a web UI.

> Part of [Artilence Tasks](https://github.com/hassaanch23/Artilence-Tasks).
> Phase 6 was a menu of advanced projects; this is the **AI Code Review Bot** option.
> Phases 1–5 live on their own branches.

---

## Project structure

```
├── .env.example / .gitignore / requirements.txt / README.md
├── examples/
│   └── sample_code.py       # a deliberately flawed file to demo the reviewer
├── src/
│   ├── config.py / llm.py
│   ├── schema.py            # Pydantic Finding + ReviewResult (the structured output)
│   ├── reviewer.py          # review a whole file  -> ReviewResult
│   ├── diff_review.py       # review a `git diff`  -> ReviewResult (PR-reviewer mode)
│   └── cli.py               # terminal entry point (+ CI exit code)
└── app.py                   # Streamlit: upload OR paste code -> findings table
```

---

## How it works

```
 source file / git diff
        │  (number the lines so the model can cite them)
        ▼
   prompt + ReviewResult schema
        │  llm.with_structured_output(ReviewResult)
        ▼
   the model is FORCED to return typed data:
        { summary, findings: [ {line, severity, category, issue, suggestion}, ... ] }
        │  sort by severity (high → low)
        ▼
   printed table / Streamlit expanders / CI exit code
```

The key move is **structured output**: we never parse prose. We hand the model the
`ReviewResult` schema and it must fill it in — so we get reliable, machine-usable data.

---

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your OpenAI key
```

Only `OPENAI_API_KEY` is required — no other services or downloads.

---

## Run

```bash
# Review a whole file
python -m src.cli examples/sample_code.py

# Review only what changed (real pull-request review)
python -m src.cli --diff              # current working-tree diff
python -m src.cli --diff main         # diff against a base ref

# Web UI: upload a file OR paste code, get a findings table
streamlit run app.py
```

The CLI **exits with code 1 when any HIGH-severity finding is present**, so you can
drop it into a CI step to fail a build on serious issues.

---

## Concept notes

### 1. Structured output — the core skill
Earlier phases asked the model for prose and read it as text. A review needs *data*:
a list of findings you can sort, count, colour, and gate a build on. LangChain's
`llm.with_structured_output(ReviewResult)` binds a **Pydantic schema** to the model, so
the reply is guaranteed to match `ReviewResult` — no fragile regex parsing. Each field's
`description` doubles as an instruction the model reads (e.g. *"1-based line number"*).

### 2. Two review modes
- **Whole file** (`reviewer.py`) — lines are numbered before sending, so findings cite
  exact locations. Good for auditing an existing file.
- **Git diff** (`diff_review.py`) — reviews only the changed (`+`) lines, exactly like a
  human PR reviewer. This is the realistic mode: you don't re-review the whole codebase
  on every commit, only the delta.

### 3. Deterministic reviews
`get_llm(temperature=0.0)`: the same code should always yield the same findings. A review
that changes each run is untrustworthy.

### 4. Fits into CI
Because the output is typed, the CLI can compute an **exit code** from it (1 on any HIGH
finding). That's what turns a chatbot into a *tool* a pipeline can rely on.

---

## Why the code is structured this way  *(talking points for PR review)*

| Decision | We chose | Instead of | Why |
|---|---|---|---|
| Output | **`with_structured_output` + Pydantic** | Prompt for text, regex-parse it | Typed, reliable findings you can sort/count/gate on. |
| Modes | **File *and* diff** | File only | Diff review is how real PRs work — review the delta, not everything. |
| Line accuracy | **Number lines in the prompt** | Hope the model counts | The model cites the number we gave it → accurate locations. |
| Determinism | **temperature 0** | Default randomness | Same code → same review. |
| CI story | **Exit 1 on HIGH** | Print and always exit 0 | Lets the bot *fail a build*, not just talk. |
| Deps | **OpenAI only** | Extra services | Runs anywhere with one key; nothing to host. |

---

## How this builds on Phases 1–5

| Phase | Reused here |
|---|---|
| **1–2** | The `config.py` / `llm.py` factory + `.env` discipline. |
| **3** | The idea of the LLM doing a *task* over your input, not just chatting. |
| **5** | Structured **Pydantic** models (there for the API request/response; here for the review itself). |

Phase 6's new idea is **constrained, typed output** — the step from "LLM that talks" to
"LLM that returns data another program can act on."

---

## Results & Findings

**Reviewing the flawed `examples/sample_code.py`** (`python -m src.cli examples/sample_code.py`):
```
6 finding(s), most severe first:
🔴 [high/security]        line 12: SQL injection via unsanitized user input
🔴 [high/security]        line 16: hardcoded API key in source
🟡 [medium/maintainability] line 19: mutable default argument
🟡 [medium/bug]           line 23: potential ZeroDivisionError on empty list
🟡 [medium/maintainability] line 28: file handle never closed
🟡 [medium/maintainability] line 29: bare `except` hides errors
(exit code: 1)   ← HIGH findings present, so CI would fail
```

**Diff mode** flags only what a change introduces — e.g. a diff that adds
`print("password attempt:", pw)` and `ADMIN_TOKEN = "hunter2"` is caught as two
`high/security` findings, while the unchanged lines are ignored.

### Headline takeaway
The leap this phase demonstrates is **structured output**: the same model that wrote
prose in Phase 1 now returns a validated list of findings with severities and line
numbers — data a terminal, a CI pipeline, or a UI can act on directly. That's what makes
an LLM a *tool*, not just a chatbot.

---

## Limitations & responsible use

This bot is an **assistant, not an oracle** — it runs on an LLM, so it will:
- **flag false positives** (things that aren't real problems),
- **miss real issues** (false negatives), and
- **miscalibrate severity** from time to time.

Treat every finding as a **suggestion for a human to verify**, not a verdict — and note
the CLI's exit-code gate is only as trustworthy as the findings behind it. This was proven
on the bot's *own* `cli.py`: it reported two false positives (a length-guarded index access
it thought could `IndexError`; a usage string it called a "high" bug) **and** missed a real
one (a raw traceback on a non-existent file path — since fixed). The reviewer prompt is
tuned to be conservative, and a stronger model (e.g. `gpt-4o` instead of `gpt-4o-mini`)
reduces this further, but no LLM reviewer eliminates it.

---

## Reproducing this environment

Tested with **Python 3.12.2**, `langchain-openai==1.3.3`, `streamlit==1.58.0`,
`python-dotenv==1.2.2` (Pydantic ships with `langchain-core`). The diff reviewer uses the
stdlib `subprocess`. Run `pip freeze > requirements-lock.txt` in a clean venv for a lock.
