"""Loads settings (OpenAI, optional Google Search, DB path) from the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- OpenAI (chat model + embeddings for vector memory) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---- Google Custom Search (OPTIONAL) ----
# If BOTH are set, web_search uses the official Google Search API.
# If either is missing, it falls back to keyless DuckDuckGo (no key needed).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# ---- Long-term memory (SQLite) ----
# One file on disk; swap for PostgreSQL by changing this + store.py's connect().
DB_PATH = os.getenv("DB_PATH", "research_assistant.db")


def google_search_enabled() -> bool:
    """True only when BOTH Google credentials are present."""
    return bool(GOOGLE_API_KEY and GOOGLE_CSE_ID)
