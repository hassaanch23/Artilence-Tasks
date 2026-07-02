"""Build, save and load a FAISS vector index from document chunks.

FAISS (Facebook AI Similarity Search) stores each chunk's embedding and, given a
query embedding, finds the nearest chunks in milliseconds. We save the index to
disk so a PDF is embedded ONCE; later queries just load it -- no repeat embedding
cost. Swap FAISS for Pinecone/Chroma by changing only this file.
"""

import os

from langchain_community.vectorstores import FAISS

from src import config
from src.embeddings import get_embeddings


def build_index(chunks):
    """Embed the chunks and return an in-memory FAISS index."""
    return FAISS.from_documents(chunks, get_embeddings())


def save_index(index, index_dir: str | None = None) -> str:
    """Persist a FAISS index to disk; returns the directory it was saved to."""
    path = index_dir or config.INDEX_DIR
    index.save_local(path)
    return path


def load_index(index_dir: str | None = None):
    """Load a previously saved FAISS index from disk."""
    path = index_dir or config.INDEX_DIR
    # allow_dangerous_deserialization: FAISS restores via pickle. Safe here because
    # WE created this file. Never load an index from an untrusted source.
    return FAISS.load_local(
        path, get_embeddings(), allow_dangerous_deserialization=True
    )


def index_exists(index_dir: str | None = None) -> bool:
    """True if a saved index is present at the given directory."""
    path = index_dir or config.INDEX_DIR
    return os.path.exists(os.path.join(path, "index.faiss"))
