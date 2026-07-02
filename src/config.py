"""Loads settings (OpenAI, embeddings, chunking, FAISS index path) from .env."""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- OpenAI (chat model + embeddings) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---- Retrieval / chunking ----
# CHUNK_SIZE  = characters per chunk we embed. Too big -> the relevant sentence is
#               diluted; too small -> it loses surrounding context. ~1000 is a good start.
# CHUNK_OVERLAP = characters shared between neighbours, so a sentence spanning a
#                 boundary is not cut in half.
# TOP_K       = how many chunks to retrieve and feed to the model per question.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# ---- FAISS index (saved to disk so a PDF is embedded only once) ----
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
