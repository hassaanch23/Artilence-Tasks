import sys

from src.llm_client import get_client

SYSTEM_PROMPT = "You are a friendly, concise assistant. Answer in one or two sentences."


class ChatBot:
    def __init__(self, client=None, system_prompt=SYSTEM_PROMPT):
        self.client = client or get_client()
        self.system_prompt = system_prompt
        self.history = []  # list of (speaker, text) tuples == the "memory"

    def _build_prompt(self, user_message: str) -> str:
        """Turn the system prompt + all past turns + the new message into one prompt."""
        lines = [self.system_prompt, ""]
        for speaker, text in self.history:
            lines.append(f"{speaker}: {text}")
        lines.append(f"User: {user_message}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def send(self, user_message: str) -> str:
        reply = self.client.generate(self._build_prompt(user_message))
        # Save BOTH sides so the next turn remembers this exchange.
        self.history.append(("User", user_message))
        self.history.append(("Assistant", reply))
        return reply


def run_repl():
    """Interactive chat loop."""
    bot = ChatBot()
    print(f"Chatbot ready (model={bot.client.model}). Type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.strip().lower() in {"exit", "quit"}:
            print("Bye!")
            break
        print("Bot:", bot.send(user), "\n")


def demo():
    """Scripted run that proves memory works, without needing stdin."""
    bot = ChatBot()
    print(f"(model={bot.client.model})  --- scripted demo ---\n")
    for message in [
        "Hi, my name is Sam.",
        "What is the capital of France?",
        "What is my name?",   # only answerable if memory works
    ]:
        print("You:", message)
        print("Bot:", bot.send(message), "\n")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        run_repl()
