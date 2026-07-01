"""
llm_client.py
-------------
The swappable LLM wrapper — the heart of this project.

THE BIG IDEA (defend this in the PR):
  Every part of our app talks to LLMs through ONE small interface: `LLMClient`.
  It never imports `openai` or `transformers` directly. So swapping OpenAI for
  Hugging Face (or adding Claude later) means adding ONE class here and changing
  nothing else. This is the "program to an interface, not an implementation"
  principle.

WHY the imports are inside the methods ("lazy imports"):
  If we imported `openai` and `torch` at the top, a user would need BOTH
  installed just to use one. By importing inside each class, you only need the
  library for the provider you actually use. Great for the "no key yet" case:
  the OpenAI code exists but never forces you to install/configure anything.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from src import config


class LLMClient(ABC):
    """Common contract every provider must implement."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Take a text prompt, return the model's text answer."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Talks to OpenAI's API. Needs OPENAI_API_KEY in your .env."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or config.OPENAI_MODEL
        self._api_key = api_key or config.OPENAI_API_KEY
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your .env file, "
                "or use the 'huggingface' provider which needs no key."
            )
        from openai import OpenAI  # lazy import
        self._client = OpenAI(api_key=self._api_key)

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 256) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class HuggingFaceClient(LLMClient):
    """Runs an open-source model locally. No API key, no per-call cost."""

    def __init__(self, model: str | None = None):
        self.model = model or config.HF_MODEL
        from transformers import pipeline  # lazy import (transformers + torch)
        # flan-t5 is a text2text (seq2seq) instruction model.
        self._pipe = pipeline("text2text-generation", model=self.model)

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        output = self._pipe(prompt, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"].strip()


def get_client(provider: str | None = None) -> LLMClient:
    """
    Factory: hand back the right client for the chosen provider.
    The rest of the app calls THIS and never constructs clients directly.
    """
    provider = (provider or config.DEFAULT_PROVIDER).lower()
    if provider == "openai":
        return OpenAIClient()
    if provider in ("huggingface", "hf"):
        return HuggingFaceClient()
    raise ValueError(
        f"Unknown provider: {provider!r}. Use 'openai' or 'huggingface'."
    )
