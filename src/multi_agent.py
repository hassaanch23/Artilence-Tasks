"""Demo: a MULTI-AGENT 'supervisor' that routes a question to a specialist agent.

The main app (support.py) uses ONE agent with three tools, which is all this project
needs. This module shows the alternative the brief highlights: several task-specific
agents coordinated by a supervisor. Here a lightweight router LLM picks the
specialist, then that specialist -- its own agent, with its own persona and its own
subset of tools -- answers.

    python -m src.multi_agent

Why show both? A single tool-using agent is simpler and has fewer failure modes; a
supervisor + specialists scales better when each role needs a distinct prompt,
different tools, or its own guardrails. Same building block (`create_agent`), wired
two ways.
"""

from langchain.agents import create_agent

from src.llm import get_llm
from src.tools import get_stock, get_weather, search_faqs


def _specialist(system_prompt: str, tools):
    return create_agent(model=get_llm(), tools=tools, system_prompt=system_prompt)


# Two task-specific specialists, each with only the tools it needs.
SPECIALISTS = {
    "product": _specialist(
        "You are a product-support specialist. Use search_faqs to answer questions "
        "about the company's products, orders, returns, warranty, and policies.",
        [search_faqs],
    ),
    "data": _specialist(
        "You are a live-data specialist. Use get_weather and get_stock to answer "
        "real-time weather and stock-price questions.",
        [get_weather, get_stock],
    ),
}

_ROUTER_PROMPT = (
    "Route the user's question to exactly one specialist. Reply with ONE word only: "
    "'product' for anything about the company, its products, orders, or policies; "
    "'data' for real-time weather or stock questions.\n\nQuestion: {q}"
)


def route(question: str) -> str:
    """Supervisor: ask a cheap, deterministic LLM which specialist should handle this."""
    reply = get_llm(temperature=0).invoke(_ROUTER_PROMPT.format(q=question))
    choice = str(reply.content).lower()  # .content can be str|list; str() keeps it safe
    return "data" if "data" in choice else "product"


def ask(question: str):
    """Return (specialist_name, answer)."""
    name = route(question)
    out = SPECIALISTS[name].invoke({"messages": [{"role": "user", "content": question}]})
    return name, out["messages"][-1].content


if __name__ == "__main__":
    for q in [
        "How long is the warranty on the Aurora X1?",
        "What's the weather in Tokyo right now?",
    ]:
        specialist, answer = ask(q)
        print(f"You: {q}")
        print(f"[supervisor routed to: {specialist} specialist]")
        print(f"Bot: {answer}\n")
