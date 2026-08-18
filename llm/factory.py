from __future__ import annotations

from .anthropic_client import AnthropicClient
from .base import LLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

PROVIDERS = {"anthropic", "openai", "openai-compatible", "ollama"}


class ProviderError(ValueError):
    pass


def get_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    extra_body: dict | None = None,
) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key, timeout=timeout)
    if provider == "openai":
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url, timeout=timeout, extra_body=extra_body)
    if provider == "openai-compatible":
        if not base_url:
            raise ProviderError("--base-url is required for provider 'openai-compatible'")
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url, timeout=timeout, extra_body=extra_body)
    if provider == "ollama":
        return OllamaClient(model=model, base_url=base_url, timeout=timeout)
    raise ProviderError(f"unknown provider '{provider}' (expected one of {sorted(PROVIDERS)})")
