"""Anthropic (Claude) backend, via API key."""

from __future__ import annotations

import os

import anthropic

from .base import ChatResponse, LLMClient, Message, ToolCall

MAX_TOKENS = 4096


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None, timeout: float | None = None):
        super().__init__(model)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        # `timeout=None` here means "unset" (use the SDK's own default) -- passing it through
        # literally would instead *disable* timeouts entirely (httpx treats None as infinite).
        kwargs = {"timeout": timeout} if timeout is not None else {}
        self._client = anthropic.Anthropic(api_key=key, **kwargs)

    def chat(self, messages: list[Message], tools: list[dict], system: str) -> ChatResponse:
        native = []
        for m in messages:
            if m.role == "user":
                native.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                native.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                native.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                        ],
                    }
                )

        anthropic_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]

        # Streamed rather than blocking: long-running requests (extended thinking, slow
        # tool-heavy turns) are more reliable over a streaming connection than one held
        # open for a single non-streaming response.
        with self._client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=native,
            tools=anthropic_tools,
        ) as stream:
            resp = stream.get_final_message()

        content = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        usage = None
        if resp.usage is not None:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                # prompt-caching fields -- absent (None) on models/API versions that don't report them
                "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", None),
                "cache_creation_tokens": getattr(resp.usage, "cache_creation_input_tokens", None),
            }

        return ChatResponse(content=content, tool_calls=tool_calls, stop_reason=resp.stop_reason, usage=usage)
