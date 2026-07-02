"""Builds the embedding model (text -> vector) for the FAQ knowledge base.

Single swap point (as in Phase 4): default OpenAI, one-line switch to a local
Hugging Face model. The SAME model must embed the FAQs and the queries.
"""

from langchain_openai import OpenAIEmbeddings

from src import config


def get_embeddings():
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing. Add it to your .env file.")
    return OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY)

    # ---- Local, no-key Hugging Face alternative ----
    # pip install langchain-huggingface sentence-transformers
    # from langchain_huggingface import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
