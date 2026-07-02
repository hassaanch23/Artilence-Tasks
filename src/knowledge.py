"""FAQ knowledge base: load the FAQ file, embed it into FAISS (the Phase 4 RAG).

This is the 'answer FAQs using an LLM + vector database' half of the project. The
FAQ markdown is split per question, embedded once, and saved to disk; the support
agent's `search_faqs` tool retrieves from it.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.embeddings import get_embeddings


def _load_faq_text() -> str:
    with open(config.FAQ_PATH, encoding="utf-8") as f:
        return f.read()


def _split(text: str):
    """One Document per FAQ entry.

    Each FAQ entry begins with a '## ' heading, so we split on those headings to
    guarantee every chunk is exactly one self-contained Q&A -- which makes retrieval
    precise (a query hits the right entry, not a blob of five). Size-based splitting
    would merge several short entries into one chunk and blur their embeddings.
    A rare over-long entry is still size-chunked as a safety net.
    """
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    docs = []
    for entry in text.split("\n## ")[1:]:  # [1:] drops the leading title block
        entry = entry.strip()
        if not entry:
            continue
        content = "## " + entry
        if len(content) <= config.CHUNK_SIZE:
            docs.append(Document(page_content=content))
        else:
            docs.extend(splitter.create_documents([content]))
    return docs


def build_index():
    """Embed the FAQ chunks and return an in-memory FAISS index."""
    return FAISS.from_documents(_split(_load_faq_text()), get_embeddings())


def save_index(index, index_dir: str | None = None) -> str:
    path = index_dir or config.INDEX_DIR
    index.save_local(path)
    return path


def load_index(index_dir: str | None = None):
    path = index_dir or config.INDEX_DIR
    # allow_dangerous_deserialization: the .pkl is unpickled; safe because we built it.
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)


def index_exists(index_dir: str | None = None) -> bool:
    path = index_dir or config.INDEX_DIR
    return os.path.exists(os.path.join(path, "index.faiss"))


def get_or_build_index():
    """Load the saved FAQ index, or build + save it on first use."""
    if index_exists():
        return load_index()
    index = build_index()
    save_index(index)
    return index


if __name__ == "__main__":
    idx = get_or_build_index()
    hits = idx.as_retriever(search_kwargs={"k": config.TOP_K}).invoke("how long is the warranty?")
    print(f"retrieved {len(hits)} FAQ chunks for 'warranty':\n")
    for h in hits:
        print("-", h.page_content[:100].replace("\n", " "), "...")
