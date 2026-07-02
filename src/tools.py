"""Tools the agent can call: web search (Google or DuckDuckGo) and a calculator.

A "tool" is just a function the LLM is allowed to call. The @tool decorator turns
a plain Python function into a LangChain tool, and its NAME, type hints and
DOCSTRING become the schema the model reads to decide *when* and *how* to call it.
So the docstrings below are not documentation for us -- they are part of the
prompt the model sees. That is why they are written to address the model.
"""

from __future__ import annotations

import ast
import json
import operator
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from src import config

SEARCH_MAX_RESULTS = 5


# ---- web search: real Google API when keyed, keyless DuckDuckGo otherwise ----

def _google_search(query: str, num: int) -> list[dict]:
    """Call Google's Custom Search JSON API directly (stdlib only, no extra deps)."""
    params = urllib.parse.urlencode(
        {"key": config.GOOGLE_API_KEY, "cx": config.GOOGLE_CSE_ID, "q": query, "num": num}
    )
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    return [
        {"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in data.get("items", [])
    ]


def _duckduckgo_search(query: str, num: int) -> list[dict]:
    """Keyless fallback: DuckDuckGo via the `ddgs` package (no API key needed)."""
    from ddgs import DDGS  # lazy import

    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=num)
    return [
        {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
        for h in hits
    ]


@tool
def web_search(query: str) -> str:
    """Search the web for current or factual information on a topic, person, or event.
    Use this whenever the answer depends on up-to-date facts you are unsure about.
    Returns a numbered list of result titles, URLs and snippets."""
    try:
        if config.google_search_enabled():
            results, source = _google_search(query, SEARCH_MAX_RESULTS), "Google"
        else:
            results, source = _duckduckgo_search(query, SEARCH_MAX_RESULTS), "DuckDuckGo"
    except Exception as e:  # return the error so the agent can react, not crash
        return f"web_search failed: {e}"

    if not results:
        return "No results found."
    lines = [f"(source: {source})"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n".join(lines)


# ---- calculator: a safe second tool the agent can choose ----

_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Evaluate ONLY arithmetic on numbers -- no names, calls, or attribute access."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if type(node.op) is ast.Pow and (abs(right) > 1000 or abs(left) > 10**6):
            raise ValueError("exponent too large")  # guard against giant-number DoS
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 5)'.
    Use this for exact math instead of computing it in your head."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception:
        return "calculator error: only numbers and + - * / // % ** are allowed."


# The tool list handed to the agent.
TOOLS = [web_search, calculator]


if __name__ == "__main__":
    print("Google search enabled:", config.google_search_enabled())
    print("\n--- web_search('who won the 2022 FIFA World Cup') ---")
    print(web_search.invoke("who won the 2022 FIFA World Cup"))
    print("\n--- calculator('2 ** 10 + 24') ---")
    print(calculator.invoke("2 ** 10 + 24"))
