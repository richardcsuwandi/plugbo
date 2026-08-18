"""OpenAI backend, via API key. Also serves any OpenAI-compatible endpoint
(vLLM, Together, Groq, LM Studio, ...) when constructed with a `base_url`.
"""

from __future__ import annotations

import json
import os
import sys
import time

import openai
from openai import OpenAI

from .base import ChatResponse, LLMClient, Message, ToolCall

VERBOSE = os.environ.get("SARA_VERBOSE", "0") != "0"

# Some OpenAI-compatible gateways (observed with ModelScope) drop a streaming
# connection mid-response under load, raised by the SDK as APIConnectionError/
# APITimeoutError -- e.g. "peer closed connection without sending complete
# message body (incomplete chunked read)". `history` in sara/agent.py's
# run_campaign is only appended to *after* chat() returns, so a mid-stream
# failure here hasn't mutated any caller state yet -- safe to just resend the
# identical request rather than fail the whole multi-hour campaign.
MAX_STREAM_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        extra_body: dict | None = None,
    ):
        super().__init__(model)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not key and not base_url:
            raise ValueError("OPENAI_API_KEY is not set")
        # `timeout=None` here means "unset" (use the SDK's own default) -- passing it through
        # literally would instead *disable* timeouts entirely (httpx treats None as infinite).
        kwargs = {"timeout": timeout} if timeout is not None else {}
        self._client = OpenAI(api_key=key or "unused", base_url=base_url, **kwargs)
        # Extra fields merged into every request body -- e.g. `{"enable_thinking": False}` to
        # turn off a Qwen3/DashScope-style model's reasoning pass, which is what makes each
        # call slow (streams hundreds of hidden "thinking" chunks before the visible content).
        self._extra_body = extra_body

    def chat(self, messages: list[Message], tools: list[dict], system: str) -> ChatResponse:
        native = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                native.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                entry: dict = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ]
                native.append(entry)
            elif m.role == "tool":
                native.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

        if VERBOSE:
            print(
                f"[llm] -> {self.model}: request sent ({len(native)} messages, {len(openai_tools)} tools)",
                file=sys.stderr,
                flush=True,
            )
        t0 = time.monotonic()

        # Streamed rather than blocking: some OpenAI-compatible gateways (observed with
        # ModelScope) drop the connection if a slow/"thinking" model holds a non-streaming
        # request open too long. Streaming keeps the connection alive with periodic chunks.
        kwargs = {"extra_body": self._extra_body} if self._extra_body else {}

        content = ""
        tool_call_acc: dict[int, dict] = {}
        finish_reason = ""
        chunk_count = 0
        first_chunk_at: float | None = None
        usage = None

        for attempt in range(MAX_STREAM_RETRIES + 1):
            try:
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=native,
                    tools=openai_tools,
                    stream=True,
                    # Standard OpenAI-protocol option (widely supported by OpenAI-compatible
                    # gateways, including ModelScope/DashScope): without it, a streaming response
                    # never reports token usage at all -- only non-streaming responses do by default.
                    stream_options={"include_usage": True},
                    **kwargs,
                )

                for chunk in stream:
                    chunk_count += 1
                    if VERBOSE and first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                        print(f"[llm]    first chunk after {first_chunk_at - t0:.1f}s", file=sys.stderr, flush=True)
                    # The usage-bearing chunk (stream_options.include_usage) is typically the last
                    # one and carries an EMPTY `choices` list -- check before the `not chunk.choices`
                    # guard below, or it's silently skipped and usage is lost.
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        cached = getattr(getattr(chunk_usage, "prompt_tokens_details", None), "cached_tokens", None)
                        usage = {
                            "input_tokens": chunk_usage.prompt_tokens,
                            "output_tokens": chunk_usage.completion_tokens,
                            "cache_read_tokens": cached,
                            "cache_creation_tokens": None,  # no such concept in the OpenAI protocol
                        }
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta and delta.content:
                        content += delta.content
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            acc = tool_call_acc.setdefault(tc_delta.index, {"id": None, "name": None, "arguments": ""})
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["arguments"] += tc_delta.function.arguments
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                break  # stream consumed fully without a connection error
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt == MAX_STREAM_RETRIES:
                    raise
                wait = RETRY_BACKOFF_SECONDS * (2**attempt)
                if VERBOSE:
                    print(
                        f"[llm] transient error ({e}), retry {attempt + 1}/{MAX_STREAM_RETRIES} in {wait:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                # A dropped mid-stream connection may have partially filled these from chunks
                # already processed above -- reset so a retried request starts from a clean slate.
                content = ""
                tool_call_acc = {}
                finish_reason = ""
                chunk_count = 0
                first_chunk_at = None
                usage = None
                time.sleep(wait)

        tool_calls = [
            ToolCall(id=acc["id"], name=acc["name"], arguments=json.loads(acc["arguments"] or "{}"))
            for _, acc in sorted(tool_call_acc.items())
        ]
        if VERBOSE:
            elapsed = time.monotonic() - t0
            names = [tc.name for tc in tool_calls]
            print(
                f"[llm] <- {self.model}: {elapsed:.1f}s total, {chunk_count} chunks, "
                f"{len(content)} content chars, tool_calls={names or 'none'}, finish_reason={finish_reason!r}, "
                f"usage={usage}",
                file=sys.stderr,
                flush=True,
            )
        return ChatResponse(content=content, tool_calls=tool_calls, stop_reason=finish_reason, usage=usage)
