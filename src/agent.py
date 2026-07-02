"""Builds the research agent using LangChain 1.x `create_agent`.

Why `create_agent` (and not `initialize_agent` / `AgentType.ZERO_SHOT_REACT...`)?
Those classic constructors every tutorial uses were REMOVED in LangChain 1.0.
`create_agent` is the modern replacement: it compiles a tool-calling, ReAct-style
agent on top of LangGraph. The model reasons, optionally calls a tool, reads the
result, and loops until it can answer -- which *is* the ReAct pattern, now driven
by the model's native tool-calling instead of brittle "Thought:/Action:" text.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.llm import get_llm
from src.tools import TOOLS

SYSTEM_PROMPT = (
    "You are an AI Research Assistant. Answer the user's question accurately and "
    "concisely. Use the web_search tool for anything needing current or factual "
    "information, and the calculator for exact arithmetic. When you use search, "
    "cite the source URLs. If the tools return nothing useful, say so honestly "
    "rather than guessing."
)


def build_agent(llm=None, checkpointer=None):
    """Return a compiled agent graph.

    The checkpointer IS the buffer memory: it stores the running message list per
    `thread_id`, so follow-up questions automatically see the earlier turns.
    """
    return create_agent(
        model=llm or get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )


if __name__ == "__main__":
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "demo"}}  # same thread == shared memory
    for q in ["What is 128 * 17?", "And what did I just ask you to compute?"]:
        print("You:", q)
        out = agent.invoke({"messages": [{"role": "user", "content": q}]}, cfg)
        print("Bot:", out["messages"][-1].content, "\n")
