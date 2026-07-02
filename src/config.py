"""Loads settings for the AI Customer Support Chatbot from the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- OpenAI (chat model + embeddings) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---- FAQ knowledge base (RAG) ----
FAQ_PATH = os.getenv("FAQ_PATH", "data/faqs.md")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
# FAQ entries are short, so smaller chunks than Phase 4's document RAG.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "3"))

# ---- Persona ----
COMPANY = os.getenv("COMPANY", "Aurora Mobility")
