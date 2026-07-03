"""The structured-output schema for a code review.

This is the heart of Phase 6's new skill: instead of asking the model for prose and
parsing it, we hand these Pydantic models to `llm.with_structured_output(...)`, and the
model is *forced* to return data in exactly this shape. Every field's description is a
hint the model reads, so the schema doubles as part of the instructions.
"""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["high", "medium", "low"]
Category = Literal["bug", "security", "performance", "style", "maintainability"]


class Finding(BaseModel):
    """One issue found in the code."""

    line: int = Field(description="1-based line number the issue refers to")
    severity: Severity = Field(description="high = likely bug/security, low = minor/style")
    category: Category
    issue: str = Field(description="what is wrong, in one sentence")
    suggestion: str = Field(description="a concrete way to fix it")


class ReviewResult(BaseModel):
    """The full review: a one-line verdict plus the list of findings."""

    summary: str = Field(description="1-2 sentence overall assessment of the code")
    findings: list[Finding] = Field(description="all issues found; empty if the code is clean")


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Most severe first, so the reader sees the important issues at the top."""
    return sorted(findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))
