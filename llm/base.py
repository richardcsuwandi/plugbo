"""Provider-agnostic message/tool representation. Each concrete client
translates this internal shape to/from its own wire format so the agent loop
never touches provider-specific dicts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" results
    tool_name: str | None = None  # set on role="tool" results


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]
    stop_reason: str
    # Normalized across providers: input_tokens, output_tokens, cache_read_tokens,
    # cache_creation_tokens. Any field the provider doesn't report is None (e.g. OpenAI-
    # protocol backends have no cache_creation concept -- that's an Anthropic-specific
    # prompt-caching mechanic). The whole dict is None if the provider reported no usage
    # at all for this turn (e.g. a gateway that ignores stream_options.include_usage).
    usage: dict | None = None


class LLMClient(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[Message], tools: list[dict], system: str) -> ChatResponse:
        """`tools` entries are `{"name", "description", "input_schema"}` (JSON Schema)."""
