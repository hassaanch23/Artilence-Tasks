"""The core reviewer: send code to the LLM and get back structured findings.

We number the lines before sending them so the model can cite accurate line numbers,
and we bind the ReviewResult schema so the reply is typed data, not prose we'd have to
parse. This is the whole-file review mode; diff_review.py handles changed-lines-only.
"""

from src.llm import get_llm
from src.schema import ReviewResult, sort_findings

# Shared calibration rules — imported by diff_review.py too, so both modes behave the
# same. These exist to cut the LLM's two biggest review failures: false positives and
# over-rated severity.
CALIBRATION = (
    " Be conservative: only report issues you are confident are genuinely real — when in "
    "doubt, leave it out, and prefer a few high-confidence findings over many speculative "
    "ones. Before reporting, read the surrounding lines and do NOT flag something already "
    "handled nearby (e.g. an index access guarded by a length check, or an error already "
    "caught). Severity: 'high' ONLY for crashes, security vulnerabilities, data loss, or "
    "clearly incorrect behaviour; 'medium' for real but non-critical bugs or maintainability "
    "problems; 'low' for minor style. Never report usage/help text, comments, or wording "
    "preferences as bugs."
)

SYSTEM = (
    "You are a meticulous senior software engineer performing a code review. Identify real, "
    "substantive problems: correctness bugs, security issues, performance problems, and "
    "maintainability concerns. For each finding give the line number, a severity, a category, "
    "a one-sentence issue description, and a concrete fix. If the code is clean, return an "
    "empty findings list." + CALIBRATION
)


def _number_lines(code: str) -> str:
    """Prefix each line with its number so the model can reference exact locations."""
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(code.splitlines(), 1))


def review_code(code: str, filename: str = "<snippet>") -> ReviewResult:
    """Review a block of source code and return structured, severity-sorted findings."""
    prompt = (
        f"{SYSTEM}\n\nReview the file `{filename}`. The lines are numbered:\n\n"
        f"{_number_lines(code)}"
    )
    # with_structured_output binds the schema: the model must return a ReviewResult.
    result = get_llm(temperature=0).with_structured_output(ReviewResult).invoke(prompt)
    result.findings = sort_findings(result.findings)
    return result


def review_file(path: str) -> ReviewResult:
    """Review a source file on disk."""
    with open(path, encoding="utf-8") as f:
        return review_code(f.read(), path)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_code.py"
    review = review_file(target)
    print("Summary:", review.summary)
    for f in review.findings:
        print(f"  [{f.severity}/{f.category}] line {f.line}: {f.issue}")
