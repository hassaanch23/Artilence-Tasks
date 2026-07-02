"""The customer-support agent: create_agent + FAQ/weather/stock tools + memory.

This reuses Phase 3's agent pattern (create_agent + InMemorySaver checkpointer) and
points it at the support tools. The agent decides per message whether to search the
FAQ knowledge base, call a live API, or just reply from the conversation so far.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src import config
from src.llm import get_llm
from src.tools import TOOLS

SYSTEM_PROMPT = (
    f"You are a friendly, professional customer-support assistant for {config.COMPANY}. "
    "For any question about the company, its products, pricing, orders, or policies, "
    "call the search_faqs tool and base your answer on what it returns. "
    "For real-time requests, use get_weather or get_stock. "
    "If the FAQs do not cover something, say you are not certain and offer to connect "
    "the customer with a human agent — do not invent policies. Keep answers concise "
    "and warm."
)


def build_agent(llm=None, checkpointer=None):
    """Return a compiled support-agent graph.

    The checkpointer is buffer memory: state is stored per `thread_id`, so each
    customer/session keeps its own conversation context.
    """
    return create_agent(
        model=llm or get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )


if __name__ == "__main__":
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "demo"}}
    for q in [
        "How long is the warranty on the battery?",   # -> search_faqs
        "What's the weather in Karachi right now?",    # -> get_weather
        "What was my very first question?",            # -> memory
    ]:
        out = agent.invoke({"messages": [{"role": "user", "content": q}]}, cfg)
        print("You:", q)
        print("Bot:", out["messages"][-1].content, "\n")
