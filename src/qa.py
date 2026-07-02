"""Q&A engine: an LCEL chain with buffer memory (a running message list)."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.llm import get_llm

SYSTEM = "You are a helpful, concise assistant. Answer clearly in 1-3 sentences."


class QASession:
    """A Q&A conversation with buffer memory (keeps the full message history)."""

    def __init__(self, llm=None):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM),
                MessagesPlaceholder("history"),  # past turns are injected here
                ("human", "{question}"),
            ]
        )
        self.chain = prompt | (llm or get_llm()) | StrOutputParser()
        self.history = []  # the buffer: list of past HumanMessage / AIMessage

    def ask(self, question: str) -> str:
        answer = self.chain.invoke({"question": question, "history": self.history})
        # Save this turn so the next call remembers it.
        self.history.append(HumanMessage(question))
        self.history.append(AIMessage(answer))
        return answer


if __name__ == "__main__":
    qa = QASession()
    for q in [
        "My name is Hassaan.",
        "What is the capital of France?",
        "What is my name?",  # only answerable if memory works
    ]:
        print("You:", q)
        print("Bot:", qa.ask(q), "\n")
