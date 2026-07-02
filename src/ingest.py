"""CLI: read a PDF, embed it, and save the FAISS index to disk.

    python -m src.ingest data/sample.pdf

Run this once per PDF; afterwards the app and `src.rag` just LOAD the saved index,
so you never pay to re-embed the same document.
"""

import sys

from src.loader import load_and_split
from src.vectorstore import build_index, save_index


def ingest(pdf_path: str, index_dir: str | None = None) -> int:
    """Load -> split -> embed -> save. Returns the number of chunks indexed."""
    chunks = load_and_split(pdf_path)
    if not chunks:
        raise ValueError(
            f"No text extracted from {pdf_path}. "
            "Is it a scanned-image PDF? Those need OCR, which is out of scope here."
        )
    path = save_index(build_index(chunks), index_dir)
    print(f"Ingested {pdf_path}: {len(chunks)} chunks -> FAISS index at '{path}/'")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.ingest <path-to-pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
