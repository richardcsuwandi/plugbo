"""Tests for sara/cli.py's `run_meta.json` bookkeeping. No network calls."""

import json

import pytest

from sara import cli
from llm.base import ChatResponse, LLMClient


class ScriptedClient(LLMClient):
    def __init__(self, responses):
        super().__init__(model="scripted")
        self._responses = list(responses)

    def chat(self, messages, tools, system):
        return self._responses.pop(0)


class Args:
    def __init__(self, **kwargs):
        self.provider = "anthropic"
        self.model = "claude-x"
        self.base_url = None
        self.api_key = None
        self.budget = 5
        self.trace = None
        self.__dict__.update(kwargs)


def test_cmd_run_writes_completed_meta(tmp_path, monkeypatch):
    context = tmp_path / "context.md"
    context.write_text("minimize y")
    workdir = tmp_path / "run1"

    client = ScriptedClient([ChatResponse(content="done", tool_calls=[], stop_reason="end_turn")])
    monkeypatch.setattr(cli, "get_client", lambda *a, **k: client)

    args = Args(context=str(context), eval="python3 eval.py", workdir=str(workdir))
    cli.cmd_run(args)

    meta = json.loads((workdir / "run_meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["kind"] == "sara"
    assert meta["provider"] == "anthropic"
    assert meta["model"] == "claude-x"
    assert meta["budget"] == 5
    assert meta["started_at"] is not None
    assert meta["ended_at"] is not None
    assert meta["n_steps"] == 1
    assert meta["final_message"] == "done"
    sidecar = json.loads((workdir / "agent_llm.json").read_text())
    assert sidecar["provider"] == "anthropic"
    assert sidecar["model"] == "claude-x"


def test_cmd_run_writes_failed_meta_on_exception(tmp_path, monkeypatch):
    context = tmp_path / "context.md"
    context.write_text("minimize y")
    workdir = tmp_path / "run2"

    class BoomClient(LLMClient):
        def __init__(self):
            super().__init__(model="boom")

        def chat(self, messages, tools, system):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(cli, "get_client", lambda *a, **k: BoomClient())

    args = Args(context=str(context), eval="python3 eval.py", workdir=str(workdir))
    with pytest.raises(RuntimeError, match="provider exploded"):
        cli.cmd_run(args)

    meta = json.loads((workdir / "run_meta.json").read_text())
    assert meta["status"] == "failed"
    assert "provider exploded" in meta["error"]
    assert meta["ended_at"] is not None
