"""Builds the LangChain chat model (ChatOpenAI) from .env config."""

from langchain_openai import ChatOpenAI

from src import config


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Return a configured ChatOpenAI model."""
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
    reply = llm.invoke("In one sentence, what is LangChain?")
    print("Reply:", reply.content)
