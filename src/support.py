"""SupportChatbot: one support agent, with memory partitioned by session_id.

This is the object both deployment targets (Streamlit app.py and FastAPI api.py)
call. Each caller passes a `session_id`; the agent's checkpointer keeps a separate
conversation per id, so many users can chat concurrently without crossing wires.
"""

from src.agent import build_agent


class SupportChatbot:
    """Thin wrapper around the support agent that routes by session_id."""

    def __init__(self):
        self.agent = build_agent()  # holds one checkpointer, keyed by thread_id

    def ask(self, message: str, session_id: str = "default") -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": session_id}},
        )
        return result["messages"][-1].content


if __name__ == "__main__":
    bot = SupportChatbot()
    # Two separate sessions prove memory is isolated per session_id.
    print("You (s1):", "Hi, what's your return policy?")
    print("Bot:", bot.ask("Hi, what's your return policy?", session_id="s1"), "\n")
    print("You (s1):", "How many days did you say I have?")
    print("Bot:", bot.ask("How many days did you say I have?", session_id="s1"), "\n")
    print("You (s2):", "What did I just ask you?")  # different session -> no memory of s1
    print("Bot:", bot.ask("What did I just ask you?", session_id="s2"))
