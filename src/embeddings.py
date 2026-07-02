"""Builds the embedding model (text -> vector). The one place to swap providers.

The brief says embeddings can come from OpenAI OR Hugging Face. We default to
OpenAI (fast, high quality, and we already use its key). Everything downstream
only calls the standard embedding interface, so switching to a local Hugging Face
model is a one-line change here and NOTHING else in the app changes.
"""

from langchain_openai import OpenAIEmbeddings

from src import config


def get_embeddings():
    """Return the embedding model used for BOTH indexing and querying.

    IMPORTANT: the same model must embed the documents and the questions. Embed
    them with different models and the vectors live in different spaces, so
    similarity search becomes meaningless.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing. Add it to your .env file.")
    return OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY)

    # ---- Local, no-key Hugging Face alternative ----
    # pip install langchain-huggingface sentence-transformers
    # from langchain_huggingface import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
