"""Vector-based memory: recall past Q&A by MEANING, not just recency.

Buffer memory feeds back the last N turns; in a long chat that misses relevant
*older* context. Vector memory embeds every past exchange and, for a new
question, retrieves the semantically closest ones -- so "what did I say about my
budget?" can surface a turn from 50 messages ago. We use OpenAIEmbeddings + an
in-memory vector store (no extra infrastructure); swap in FAISS/Chroma to persist
the vectors at scale.
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from src import config


class VectorMemory:
    """Semantic recall over past exchanges."""

    def __init__(self, embeddings=None):
        self.store = InMemoryVectorStore(
            embeddings
            or OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY)
        )

    def add(self, question: str, answer: str) -> None:
        # Store the whole exchange as one document so recall returns full context.
        self.store.add_texts([f"Q: {question}\nA: {answer}"])

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Return the k most semantically similar past exchanges."""
        return [d.page_content for d in self.store.similarity_search(query, k=k)]

    def load(self, pairs) -> None:
        """Warm the store from persisted (question, answer) rows (past sessions)."""
        texts = [f"Q: {q}\nA: {a}" for q, a in pairs]
        if texts:
            self.store.add_texts(texts)


if __name__ == "__main__":
    mem = VectorMemory()
    mem.add("My favorite language is Python.", "Noted!")
    mem.add("The capital of France is Paris.", "Correct.")
    print("recall('what language do I like?'):")
    for hit in mem.recall("what programming language do I like?"):
        print(" -", hit.replace("\n", " | "))
