"""Load a PDF and split it into overlapping chunks ready for embedding.

Why split at all? Embedding + retrieval work best on small, focused passages. A
whole page is too coarse (the one relevant sentence gets diluted among hundreds of
others); a single sentence is too fine (it loses surrounding context). ~1000-char
chunks with a little overlap is the sweet spot.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def load_pdf(path: str):
    """Read a PDF into a list of LangChain Documents, one per page.

    Each Document carries the page text plus metadata (source path + page number).
    We later surface that page number as a citation.
    """
    return PyPDFLoader(path).load()


def split_documents(documents):
    """Split page-level Documents into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Prefer to break on paragraph, then line, then sentence, then word --
        # so a chunk boundary lands at a natural pause, not mid-word.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(path: str):
    """Convenience: PDF path -> ready-to-embed chunks (Documents)."""
    return split_documents(load_pdf(path))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    chunks = load_and_split(path)
    print(f"{path}: {len(chunks)} chunks")
    if chunks:
        first = chunks[0]
        print("first chunk page :", first.metadata.get("page"))
        print("first chunk text :", first.page_content[:200].replace("\n", " "), "...")
