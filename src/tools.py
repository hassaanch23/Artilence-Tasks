"""Tools the support agent can call:

  - search_faqs : answer product/policy questions from the FAQ knowledge base (RAG)
  - get_weather : live weather via Open-Meteo (keyless public API)
  - get_stock   : live stock quote via Yahoo Finance (keyless public API)

As in Phase 3, each tool's docstring is the description the model reads to decide
when to call it. The two live tools cover the brief's 'fetch real-time data from
APIs' requirement -- and, unlike the model's training data, they return CURRENT info.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from src import config
from src.knowledge import get_or_build_index

# Build/load the FAQ index once, lazily (the first FAQ question triggers it).
_faq_index = None


def _faqs():
    global _faq_index
    if _faq_index is None:
        _faq_index = get_or_build_index()
    return _faq_index


def _http_json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


@tool
def search_faqs(query: str) -> str:
    """Search the company's FAQ / knowledge base to answer customer questions about
    products, pricing, orders, shipping, returns, warranty, and policies. Use this
    FIRST for anything about the company or its products."""
    docs = _faqs().as_retriever(search_kwargs={"k": config.TOP_K}).invoke(query)
    if not docs:
        return "No matching FAQ was found."
    return "\n\n".join(d.page_content for d in docs)


@tool
def get_weather(city: str) -> str:
    """Get the CURRENT weather for a city using a live API. Use for real-time weather
    questions like 'what's the weather in London?'."""
    try:
        geo = _http_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": city, "count": 1})
        )
        if not geo.get("results"):
            return f"Could not find a location named {city!r}."
        loc = geo["results"][0]
        data = _http_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                }
            )
        )
        c = data["current"]
        return (
            f"Current weather in {loc['name']}, {loc.get('country', '')}: "
            f"{c['temperature_2m']}°C, humidity {c['relative_humidity_2m']}%, "
            f"wind {c['wind_speed_10m']} km/h."
        )
    except Exception as e:  # keep the agent alive; let it read the error
        return f"weather lookup failed: {e}"


@tool
def get_stock(symbol: str) -> str:
    """Get the CURRENT price of a stock ticker (e.g. 'AAPL', 'TSLA') using a live API."""
    try:
        data = _http_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}",
            headers={"User-Agent": "Mozilla/5.0"},  # Yahoo requires a UA header
        )
        m = data["chart"]["result"][0]["meta"]
        return (
            f"{m.get('symbol', symbol)} is trading at "
            f"{m.get('regularMarketPrice')} {m.get('currency', '')}."
        )
    except Exception as e:
        return f"stock lookup failed for {symbol!r}: {e}"


# The tools handed to the support agent.
TOOLS = [search_faqs, get_weather, get_stock]


if __name__ == "__main__":
    print("--- search_faqs('warranty') ---")
    print(search_faqs.invoke("how long is the warranty?"))
    print("\n--- get_weather('London') ---")
    print(get_weather.invoke("London"))
    print("\n--- get_stock('AAPL') ---")
    print(get_stock.invoke("AAPL"))
