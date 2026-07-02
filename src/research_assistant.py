"""AI Research Assistant: an agent (search + calculator) wired to three memories.

  - buffer memory  -> the agent's checkpointer, per thread_id (immediate follow-ups)
  - long-term      -> SQLite (survives restarts)
  - vector memory  -> semantic recall of older / cross-session context

This is the Phase 3 hands-on project: ask a question, the agent decides whether to
search the web or calculate, and every exchange is remembered for context-aware
answers -- even across restarts.
"""

from src.agent import build_agent
from src.memory import VectorMemory
from src.store import QAStore


class ResearchAssistant:
    """One conversation thread, combining the agent with all three memories."""

    def __init__(self, thread_id: str = "default", store=None, vector_memory=None):
        self.thread_id = thread_id
        self.agent = build_agent()  # includes its own buffer-memory checkpointer
        self.store = store or QAStore()
        self.vmem = vector_memory or VectorMemory()
        # Warm vector memory from anything persisted in previous runs (long-term).
        self.vmem.load(self.store.all_pairs())

    def ask(self, question: str) -> str:
        # 1) Pull semantically relevant older context (may be from a past session).
        recalled = self.vmem.recall(question)
        if recalled:
            context = "\n".join(recalled)
            content = f"(Relevant context from earlier:\n{context}\n)\n\nQuestion: {question}"
        else:
            content = question

        # 2) Run the agent (buffer memory is handled automatically via thread_id).
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
            {"configurable": {"thread_id": self.thread_id}},
        )
        answer = result["messages"][-1].content

        # 3) Persist to long-term (SQLite) AND vector memory for next time.
        self.store.save(self.thread_id, question, answer)
        self.vmem.add(question, answer)
        return answer


if __name__ == "__main__":
    ra = ResearchAssistant(thread_id="demo")
    for q in [
        "My name is Hassaan and I'm researching solar energy.",
        "Search the web for a recent advance in solar panel efficiency.",
        "What is my name and what am I researching?",  # only works if memory works
    ]:
        print("You:", q)
        print("Bot:", ra.ask(q), "\n")
