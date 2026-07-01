"""
playground.py
-------------
The Prompt Engineering Playground.

It runs the SAME question through all three prompt styles against the SAME model,
so you can see exactly how the prompt style changes the answer.

Run it from the project root:
    python -m src.playground
"""

from src.llm_client import get_client
from src.prompts import Task, STYLES


def compare_styles(task: Task, question: str, client=None) -> dict:
    """Run one question through every prompt style; print + return the answers."""
    client = client or get_client()

    print("=" * 72)
    print(f"TASK    : {task.name}")
    print(f"QUESTION: {question}")
    print("=" * 72)

    answers = {}
    for style_name, build_prompt in STYLES.items():
        prompt = build_prompt(task, question)
        answer = client.generate(prompt)
        answers[style_name] = answer
        print(f"\n--- {style_name} ---")
        print(f"answer: {answer}")

    print()
    return answers


# ---- Two example tasks that show DIFFERENT strengths ----

SENTIMENT = Task(
    name="Sentiment classification",
    instruction="Classify the sentiment of the input as exactly 'positive' or 'negative'.",
    examples=[
        ("I absolutely loved this movie.", "positive"),
        ("The food was cold and completely tasteless.", "negative"),
    ],
)

MATH = Task(
    name="Arithmetic word problem",
    instruction="Solve the math word problem and give the final number.",
    examples=[
        ("Roger has 5 balls. He buys 2 more. How many balls does he have?", "7"),
    ],
)

INITIALS = Task(
    name="Custom-format extraction (initials)",
    instruction="Give the person's initials.",
    examples=[
        ("Ada Lovelace", "A.L."),
        ("Alan Turing", "A.T."),
    ],
)


def main():
    client = get_client()  # load the model ONCE, reuse for every call
    print(f"(provider={type(client).__name__}, model={client.model})\n")

    # Simple task: zero-shot is usually enough for an instruction-tuned model.
    compare_styles(SENTIMENT, "The plot was boring but the acting was great.", client)

    # Custom format: few-shot should win because only the examples show the
    # exact "A.L." dotted format the instruction alone does not specify.
    compare_styles(INITIALS, "Grace Hopper", client)

    # Multi-step reasoning: chain-of-thought forces the model to work it out.
    compare_styles(MATH, "A shop has 3 boxes with 4 apples each. How many apples in total?", client)


if __name__ == "__main__":
    main()
