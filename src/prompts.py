from dataclasses import dataclass, field


@dataclass
class Task:
    name: str                       # human label, e.g. "Sentiment classification"
    instruction: str                # what we ask the model to do
    examples: list = field(default_factory=list)  # list of (input, output) pairs


def zero_shot(task: Task, question: str) -> str:
    """Just the instruction + the question. No examples."""
    return f"{task.instruction}\n\nInput: {question}\nOutput:"


def few_shot(task: Task, question: str) -> str:
    """Instruction + a few worked examples, then the real question."""
    shots = ""
    for example_input, example_output in task.examples:
        shots += f"Input: {example_input}\nOutput: {example_output}\n\n"
    return f"{task.instruction}\n\n{shots}Input: {question}\nOutput:"


def chain_of_thought(task: Task, question: str) -> str:
    """Instruction + the question + a nudge to reason step by step first."""
    return (
        f"{task.instruction}\n\n"
        f"Input: {question}\n"
        "Let's think step by step, then give the final answer."
    )


# Registry so the playground can loop over all styles by name.
STYLES = {
    "zero-shot": zero_shot,
    "few-shot": few_shot,
    "chain-of-thought": chain_of_thought,
}
