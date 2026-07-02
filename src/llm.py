"""Builds the LangChain chat model (ChatOpenAI) from .env config."""

from langchain_openai import ChatOpenAI

from src import config


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return a configured ChatOpenAI model.

    Lower default temperature than Phase 2's chatbot: a research assistant should
    be factual and repeatable, not creative.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing. Add it to your .env file.")
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=temperature,
    )


if __name__ == "__main__":
    llm = get_llm()
    print(f"Model: {config.OPENAI_MODEL}")
    print("Reply:", llm.invoke("Say hi in five words.").content)
