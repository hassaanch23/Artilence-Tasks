"""Command-line interface for the AI Code Review Bot.

    python -m src.cli examples/sample_code.py     # review a whole file
    python -m src.cli --diff                       # review the current git diff
    python -m src.cli --diff main                  # review the diff against a base ref

Exit code is 1 when any HIGH-severity finding is present, so this can gate CI.
"""

import os
import sys

from src.diff_review import get_diff, review_diff
from src.reviewer import review_file

_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵"}


def _print(result, title: str) -> int:
    """Print a review; return the process exit code (1 if any HIGH finding)."""
    if result is None:
        print("No changes to review (empty git diff).")
        return 0
    print(f"\n=== Review: {title} ===")
    print("Summary:", result.summary)
    if not result.findings:
        print("✅ No issues found.")
        return 0
    print(f"\n{len(result.findings)} finding(s), most severe first:\n")
    for f in result.findings:
        print(f"{_ICON.get(f.severity, '•')} [{f.severity}/{f.category}] line {f.line}: {f.issue}")
        print(f"     → fix: {f.suggestion}")
    return 1 if any(f.severity == "high" for f in result.findings) else 0


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("usage: python -m src.cli <file> | --diff [base_ref]")
        sys.exit(2)

    if args[0] == "--diff":
        base = args[1] if len(args) > 1 else None
        code = _print(review_diff(get_diff(base)), f"git diff {base or ''}".strip())
    else:
        path = args[0]
        if not os.path.isfile(path):   # clean error instead of a raw traceback
            print(f"error: file not found: {path}")
            sys.exit(2)
        code = _print(review_file(path), path)
    sys.exit(code)


if __name__ == "__main__":
    main()
