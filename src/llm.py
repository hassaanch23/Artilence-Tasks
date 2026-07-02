"""Builds the LangChain chat model (ChatOpenAI) from .env config."""

from langchain_openai import ChatOpenAI

from src import config


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Return a configured ChatOpenAI model.

    Low temperature on purpose: a RAG answer should stick closely to the retrieved
    document text, not improvise.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing. Add it to your .env file.")
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=temperature,
    )


if __name__ == "__main__":
    print("Model:", config.OPENAI_MODEL)
    print("Reply:", get_llm().invoke("Say hi in five words.").content)
