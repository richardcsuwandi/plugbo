"""Provider dispatch and message-translation tests. No network calls: the
underlying SDK clients are monkeypatched.
"""

from unittest.mock import MagicMock

import pytest

from llm.base import Message, ToolCall
from llm.factory import ProviderError, get_client


class _FnDelta:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index, id=None, function=None):
        self.index = index
        self.id = id
        self.function = function


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _PromptTokensDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _OpenAIUsage:
    def __init__(self, prompt_tokens, completion_tokens, cached_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _PromptTokensDetails(cached_tokens) if cached_tokens is not None else None


def test_factory_requires_base_url_for_openai_compatible():
    with pytest.raises(ProviderError):
        get_client("openai-compatible", "some-model")


def test_factory_unknown_provider():
    with pytest.raises(ProviderError):
        get_client("bogus", "some-model")


def test_factory_anthropic_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        get_client("anthropic", "claude-x")


def test_factory_ollama_defaults_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = get_client("ollama", "llama3.1")
    assert str(client._client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_anthropic_chat_translates_tool_use(monkeypatch):
    from llm.anthropic_client import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient(model="claude-x")

    fake_block = MagicMock(type="tool_use", id="call_1", input={"cmd": "lenz status"})
    fake_block.name = "bash"  # `name=` in the MagicMock() constructor sets the mock's repr, not an attribute
    fake_resp = MagicMock(content=[fake_block], stop_reason="tool_use")

    stream_cm = MagicMock()
    stream_cm.__enter__.return_value.get_final_message.return_value = fake_resp
    stream_cm.__exit__.return_value = False
    client._client = MagicMock()
    client._client.messages.stream.return_value = stream_cm

    history = [Message(role="user", content="go")]
    tools = [{"name": "bash", "description": "run a shell command", "input_schema": {"type": "object"}}]
    result = client.chat(history, tools, system="you are sara")

    assert result.stop_reason == "tool_use"
    assert result.tool_calls == [ToolCall(id="call_1", name="bash", arguments={"cmd": "lenz status"})]
    called_kwargs = client._client.messages.stream.call_args.kwargs
    assert called_kwargs["system"] == "you are sara"
    assert called_kwargs["tools"][0]["name"] == "bash"


def test_openai_chat_translates_tool_calls(monkeypatch):
    from llm.openai_client import OpenAIClient

    client = OpenAIClient(model="gpt-x", api_key="test-key")

    chunks = [
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, id="call_1", function=_FnDelta(name="bash", arguments=""))]))]),
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, function=_FnDelta(arguments='{"cmd": "lenz status"}'))]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = chunks

    history = [Message(role="user", content="go")]
    tools = [{"name": "bash", "description": "run a shell command", "input_schema": {"type": "object"}}]
    result = client.chat(history, tools, system="you are sara")

    assert result.tool_calls == [ToolCall(id="call_1", name="bash", arguments={"cmd": "lenz status"})]
    called_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert called_kwargs["messages"][0] == {"role": "system", "content": "you are sara"}
    assert called_kwargs["tools"][0]["function"]["name"] == "bash"


def test_openai_chat_round_trips_tool_result_message(monkeypatch):
    from llm.openai_client import OpenAIClient

    client = OpenAIClient(model="gpt-x", api_key="test-key")
    chunks = [
        _Chunk([_Choice(_Delta(content="done"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
    ]
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = chunks

    history = [
        Message(role="user", content="go"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="call_1", name="bash", arguments={})]),
        Message(role="tool", tool_call_id="call_1", tool_name="bash", content='{"ok": true}'),
    ]
    result = client.chat(history, tools=[], system="sys")
    assert result.content == "done"
    sent = client._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}


def test_openai_chat_requests_and_captures_streamed_usage():
    """The usage-bearing chunk (stream_options.include_usage) carries an EMPTY
    choices list per the OpenAI protocol -- must not be skipped by the
    `if not chunk.choices: continue` guard, or usage silently vanishes.
    """
    from llm.openai_client import OpenAIClient

    client = OpenAIClient(model="gpt-x", api_key="test-key")
    chunks = [
        _Chunk([_Choice(_Delta(content="done"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
        _Chunk([], usage=_OpenAIUsage(prompt_tokens=120, completion_tokens=30, cached_tokens=80)),
    ]
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = chunks

    history = [Message(role="user", content="go")]
    result = client.chat(history, tools=[], system="sys")

    assert result.usage == {
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_read_tokens": 80,
        "cache_creation_tokens": None,
    }
    called_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert called_kwargs["stream_options"] == {"include_usage": True}


def test_openai_chat_usage_none_when_gateway_never_reports_it():
    from llm.openai_client import OpenAIClient

    client = OpenAIClient(model="gpt-x", api_key="test-key")
    chunks = [_Chunk([_Choice(_Delta(content="done"), finish_reason="stop")])]
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = chunks

    history = [Message(role="user", content="go")]
    result = client.chat(history, tools=[], system="sys")
    assert result.usage is None


def test_anthropic_chat_captures_usage(monkeypatch):
    from llm.anthropic_client import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient(model="claude-x")

    fake_usage = MagicMock(input_tokens=200, output_tokens=50, cache_creation_input_tokens=10, cache_read_input_tokens=5)
    fake_resp = MagicMock(content=[], stop_reason="end_turn", usage=fake_usage)

    stream_cm = MagicMock()
    stream_cm.__enter__.return_value.get_final_message.return_value = fake_resp
    stream_cm.__exit__.return_value = False
    client._client = MagicMock()
    client._client.messages.stream.return_value = stream_cm

    history = [Message(role="user", content="go")]
    result = client.chat(history, tools=[], system="sys")

    assert result.usage == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 10,
    }
