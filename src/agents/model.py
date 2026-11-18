from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-3.1-flash-lite"


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Google Gemini API credentials are missing. Set GOOGLE_API_KEY "
            "or GOOGLE_GENAI_API_KEY in the environment or .env file."
        )
    return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0)
