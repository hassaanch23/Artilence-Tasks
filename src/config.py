import os
from dotenv import load_dotenv

# Reads the .env file (if present) and loads its KEY=VALUE pairs into
# environment variables. Safe to call even if .env is missing.
load_dotenv()

# Which provider to use unless the caller overrides it.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")

# ---- OpenAI settings ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---- Hugging Face settings ----
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
