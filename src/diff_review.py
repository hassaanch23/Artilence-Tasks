"""Review a `git diff` — the real pull-request-reviewer mode.

Instead of a whole file, this reviews only what CHANGED, exactly like a human PR
reviewer. It runs `git diff`, hands the unified diff to the model, and asks it to
comment only on added/modified lines using the file + line info in the @@ hunk headers.
"""

import subprocess

from src.llm import get_llm
from src.reviewer import CALIBRATION
from src.schema import ReviewResult, sort_findings

SYSTEM = (
    "You are reviewing a git diff in unified format. Review ONLY the changed lines "
    "(added lines start with '+'). Report bugs, security issues, performance problems, "
    "and maintainability concerns introduced by the change. Use the file path and the "
    "line numbers from the '@@' hunk headers to set each finding's line. Do not comment "
    "on unchanged context lines. If the change looks fine, return an empty findings list."
    + CALIBRATION
)


def get_diff(base: str | None = None) -> str:
    """Return the current `git diff` (optionally against a base ref/commit)."""
    cmd = ["git", "diff"]
    if base:
        cmd.append(base)
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def review_diff(diff_text: str) -> ReviewResult | None:
    """Review a unified diff. Returns None if the diff is empty."""
    if not diff_text.strip():
        return None
    result = (
        get_llm(temperature=0)
        .with_structured_output(ReviewResult)
        .invoke(f"{SYSTEM}\n\nHere is the diff:\n\n{diff_text}")
    )
    result.findings = sort_findings(result.findings)
    return result


if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else None
    review = review_diff(get_diff(base))
    if review is None:
        print("No changes to review (empty git diff).")
    else:
        print("Summary:", review.summary)
        for f in review.findings:
            print(f"  [{f.severity}/{f.category}] line {f.line}: {f.issue}")
