"""Prompt templates and the LCEL chain for the Q&A app."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm

SYSTEM = "You are a helpful, concise assistant. Answer clearly in 1-3 sentences."


def build_prompt() -> ChatPromptTemplate:
    """Chat prompt: a fixed system instruction + a dynamic {question} variable."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "{question}"),
        ]
    )


def build_few_shot_prompt() -> ChatPromptTemplate:
    """Few-shot: worked examples baked in teach the model the exact output format."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Classify the sentiment as exactly 'positive' or 'negative'."),
            ("human", "I loved this movie."),
            ("ai", "positive"),
            ("human", "The food was cold and bland."),
            ("ai", "negative"),
            ("human", "{text}"),
        ]
    )


def build_chain(llm=None):
    """LCEL chain: fill the prompt -> call the model -> parse to a plain string."""
    llm = llm or get_llm()
    return build_prompt() | llm | StrOutputParser()


if __name__ == "__main__":
    print("=== 1) Basic template with a dynamic {question} ===")
    for m in build_prompt().invoke({"question": "What is LangChain?"}).to_messages():
        print(f"[{m.type}] {m.content}")

    print("\n=== 2) Few-shot template (examples teach the format) ===")
    for m in build_few_shot_prompt().invoke({"text": "Best purchase ever!"}).to_messages():
        print(f"[{m.type}] {m.content}")

    print("\n=== 3) Run the full LCEL chain: prompt | model | parser ===")
    answer = build_chain().invoke({"question": "What is LangChain in one sentence?"})
    print("answer:", answer)
