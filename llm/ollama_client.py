"""Local models via Ollama's OpenAI-compatible endpoint. No real API key needed."""

from __future__ import annotations

import os

from .openai_client import OpenAIClient

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaClient(OpenAIClient):
    def __init__(self, model: str, base_url: str | None = None, timeout: float | None = None):
        resolved = base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
        super().__init__(model=model, api_key="ollama", base_url=resolved, timeout=timeout)
