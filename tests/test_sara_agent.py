import json

from sara.agent import _progress_line, run_campaign
from llm.base import ChatResponse, LLMClient, ToolCall


class ScriptedClient(LLMClient):
    """Replays a fixed sequence of responses, ignoring the actual message history."""

    def __init__(self, responses: list[ChatResponse]):
        super().__init__(model="scripted")
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools, system):
        self.calls += 1
        return self._responses.pop(0)


def test_agent_executes_tool_call_then_reports(tmp_path):
    responses = [
        ChatResponse(
            content="checking status",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo hello"})],
            stop_reason="tool_use",
        ),
        ChatResponse(content="done: incumbent is x=1", tool_calls=[], stop_reason="end_turn"),
    ]
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path)

    assert result.final_message == "done: incumbent is x=1"
    assert result.n_steps == 2
    tool_entries = [e for e in result.transcript if e.get("role") == "tool"]
    assert len(tool_entries) == 1
    assert "hello" in tool_entries[0]["content"]


def test_agent_sums_usage_across_turns_and_logs_it_per_turn(tmp_path):
    responses = [
        ChatResponse(
            content="checking status",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo hi"})],
            stop_reason="tool_use",
            usage={"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": None, "cache_creation_tokens": None},
        ),
        ChatResponse(
            content="done",
            tool_calls=[],
            stop_reason="end_turn",
            usage={"input_tokens": 150, "output_tokens": 10, "cache_read_tokens": 5, "cache_creation_tokens": None},
        ),
    ]
    client = ScriptedClient(responses)
    trace_path = tmp_path / "trace.jsonl"
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, trace_path=trace_path)

    assert result.usage_total == {
        "input_tokens": 250,
        "output_tokens": 30,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 0,
    }
    assistant_entries = [json.loads(l) for l in trace_path.read_text().strip().splitlines() if json.loads(l).get("role") == "assistant"]
    assert assistant_entries[0]["usage"]["input_tokens"] == 100
    assert assistant_entries[1]["usage"]["input_tokens"] == 150


def test_agent_usage_total_is_none_when_provider_never_reports_it(tmp_path):
    responses = [ChatResponse(content="done", tool_calls=[], stop_reason="end_turn")]  # usage defaults to None
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path)
    assert result.usage_total is None


def test_agent_writes_trace_file(tmp_path):
    responses = [ChatResponse(content="ok", tool_calls=[], stop_reason="end_turn")]
    client = ScriptedClient(responses)
    trace_path = tmp_path / "trace.jsonl"
    run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, trace_path=trace_path)

    lines = trace_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    entries = [json.loads(line) for line in lines]  # each line is valid JSON
    assert entries[0]["role"] == "user" and entries[0]["content"] == "go"  # initial prompt is actually on disk
    assert isinstance(entries[0]["ts"], (int, float))
    assert entries[1]["role"] == "assistant"


def test_agent_trace_file_is_fresh_per_run(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"role": "user", "content": "stale from a previous run"}\n')

    client = ScriptedClient([ChatResponse(content="ok", tool_calls=[], stop_reason="end_turn")])
    run_campaign(client, system_prompt="sys", user_prompt="fresh prompt", sandbox=tmp_path, trace_path=trace_path)

    entries = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert entries[0]["role"] == "user" and entries[0]["content"] == "fresh prompt"
    assert not any(e.get("content") == "stale from a previous run" for e in entries)


def _write_state(path, objectives, trials):
    state = {"shelf": {"objectives": objectives}, "trials": trials}
    path.write_text(json.dumps(state))


def test_progress_line_single_objective_minimize(tmp_path):
    trials = [
        {"status": "observed", "metrics": {"y": 5.0}, "observed_at": 1},
        {"status": "observed", "metrics": {"y": 2.0}, "observed_at": 2},
        {"status": "observed", "metrics": {"y": 3.0}, "observed_at": 3},
    ]
    _write_state(tmp_path / "state.json", [{"metric": "y", "minimize": True}], trials)
    line = _progress_line(tmp_path, budget=10)
    assert line == "eval 3/10  y=3  best=2"


def test_progress_line_maximize_uses_max_as_best(tmp_path):
    trials = [
        {"status": "observed", "metrics": {"y": 1.0}, "observed_at": 1},
        {"status": "observed", "metrics": {"y": 4.0}, "observed_at": 2},
    ]
    _write_state(tmp_path / "state.json", [{"metric": "y", "minimize": False}], trials)
    line = _progress_line(tmp_path, budget=None)
    assert line == "eval 2  y=4  best=4"


def test_progress_line_multi_objective(tmp_path):
    trials = [{"status": "observed", "metrics": {"y1": 1.0, "y2": 2.0}, "observed_at": 1}]
    _write_state(tmp_path / "state.json", [{"metric": "y1", "minimize": True}, {"metric": "y2", "minimize": False}], trials)
    line = _progress_line(tmp_path, budget=20)
    assert line == "eval 1/20  (2 objectives)"


def test_progress_line_no_observations_returns_none(tmp_path):
    _write_state(tmp_path / "state.json", [{"metric": "y", "minimize": True}], [])
    assert _progress_line(tmp_path, budget=10) is None


def test_progress_line_missing_state_returns_none(tmp_path):
    assert _progress_line(tmp_path, budget=10) is None


def test_agent_prints_progress_line_on_each_new_observation(tmp_path, capsys):
    def write_trial(y):
        state_path = tmp_path / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "shelf": {"objectives": [{"metric": "y", "minimize": True}]},
            "trials": [],
        }
        state["trials"].append({"status": "observed", "metrics": {"y": y}, "observed_at": len(state["trials"]) + 1})
        state_path.write_text(json.dumps(state))

    class RecordingClient(LLMClient):
        def __init__(self):
            super().__init__(model="scripted")
            self.calls = 0

        def chat(self, messages, tools, system):
            self.calls += 1
            if self.calls <= 2:
                write_trial(10.0 - self.calls)
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(id=str(self.calls), name="bash", arguments={"cmd": "echo ok"})],
                    stop_reason="tool_use",
                )
            return ChatResponse(content="done", tool_calls=[], stop_reason="end_turn")

    run_campaign(RecordingClient(), system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=5)
    err = capsys.readouterr().err
    assert "eval 1/5" in err
    assert "eval 2/5" in err


def test_agent_nudges_when_over_budget(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"trials": [{"status": "observed"} for _ in range(10)]})
    )
    responses = [
        ChatResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo x"})],
            stop_reason="tool_use",
        ),
        ChatResponse(content="wrapping up", tool_calls=[], stop_reason="end_turn"),
    ]
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=5)

    nudges = [e for e in result.transcript if e.get("role") == "user" and "budget of 5" in e.get("content", "")]
    assert len(nudges) == 1


def test_agent_stops_at_max_steps(tmp_path):
    looping = ChatResponse(
        content="",
        tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo x"})],
        stop_reason="tool_use",
    )

    class InfiniteClient(LLMClient):
        def __init__(self):
            super().__init__(model="scripted")

        def chat(self, messages, tools, system):
            return looping

    result = run_campaign(
        InfiniteClient(), system_prompt="sys", user_prompt="go", sandbox=tmp_path, max_steps=3
    )
    assert result.n_steps == 3
    assert "max computational steps" in result.final_message


def test_read_tool_blocks_escape(tmp_path):
    from sara.tools import build_tools

    (tmp_path / "context.md").write_text("hello world")
    _, handlers = build_tools(tmp_path)
    assert handlers["read"]("context.md") == "hello world"
    assert "outside the sandbox" in handlers["read"]("../../etc/passwd")
    assert "not found" in handlers["read"]("missing.txt")


def test_bash_tool_runs_in_sandbox(tmp_path):
    from sara.tools import build_tools

    _, handlers = build_tools(tmp_path)
    out = handlers["bash"]("pwd")
    assert str(tmp_path.resolve()) in out
