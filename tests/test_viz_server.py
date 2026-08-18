import json

from viz.server import (
    _classify_lenz_calls,
    _compute_convergence,
    _duration_seconds,
    _kernel_generations,
    _reconfigurations,
    _run_kind,
    _tool_use_events,
    delete_run,
    find_runs,
    run_detail,
)


def _write_state(path, objectives, trials, events=None, surrogate="fixed", kernel_populations=None):
    pops = kernel_populations or {}
    state = {
        "space": {"x": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"}},
        "shelf": {
            "objectives": objectives,
            "constraints": [],
            "acqf": "noisy_logei",
            "acqf_params": {},
            "bounds": {},
            "surrogate": surrogate,
            "kernel_populations": pops,
            "kernel_evolution_states": {"y": {"generation": 1, "frozen": False}} if pops else {},
        },
        "trials": trials,
        "events": events or [],
    }
    path.write_text(json.dumps(state))
    return state


def test_find_runs_discovers_nested_state_files(tmp_path):
    (tmp_path / "run1").mkdir()
    _write_state(tmp_path / "run1" / "state.json", [{"metric": "y", "minimize": True}], [])
    (tmp_path / "nested" / "run2").mkdir(parents=True)
    _write_state(tmp_path / "nested" / "run2" / "state.json", [{"metric": "y", "minimize": True}], [])

    runs = find_runs(tmp_path)
    names = {r["name"] for r in runs}
    assert names == {"run1", "nested/run2"}


def test_find_runs_empty_root(tmp_path):
    assert find_runs(tmp_path / "does-not-exist") == []


def test_find_runs_reports_surrogate_and_counts(tmp_path):
    (tmp_path / "cakerun").mkdir()
    trials = [
        {"trial_id": "a", "config": {"x": 0.1}, "metrics": {"y": 1.0}, "status": "observed"},
        {"trial_id": "b", "config": {"x": 0.2}, "metrics": None, "status": "in_flight"},
    ]
    _write_state(tmp_path / "cakerun" / "state.json", [{"metric": "y", "minimize": True}], trials, surrogate="cake")

    runs = find_runs(tmp_path)
    assert runs[0]["surrogate"] == "cake"
    assert runs[0]["n_trials"] == 2
    assert runs[0]["n_observed"] == 1


def test_run_detail_blocks_path_traversal(tmp_path):
    (tmp_path / "run1").mkdir()
    _write_state(tmp_path / "run1" / "state.json", [{"metric": "y", "minimize": True}], [])
    assert run_detail(tmp_path, "../outside") is None
    assert run_detail(tmp_path, "does-not-exist") is None
    assert run_detail(tmp_path, "run1") is not None


def test_compute_convergence_minimize_tracks_running_best():
    state = {
        "shelf": {"objectives": [{"metric": "y", "minimize": True}]},
        "trials": [
            {"trial_id": "a", "metrics": {"y": 5.0}, "status": "observed", "observed_at": 1},
            {"trial_id": "b", "metrics": {"y": 2.0}, "status": "observed", "observed_at": 2},
            {"trial_id": "c", "metrics": {"y": 3.0}, "status": "observed", "observed_at": 3},
            {"trial_id": "d", "metrics": None, "status": "in_flight", "observed_at": None},
        ],
    }
    conv = _compute_convergence(state)
    assert conv["metric"] == "y"
    assert [p["best"] for p in conv["points"]] == [5.0, 2.0, 2.0]  # monotonically non-increasing


def test_compute_convergence_maximize_tracks_running_best():
    state = {
        "shelf": {"objectives": [{"metric": "y", "minimize": False}]},
        "trials": [
            {"trial_id": "a", "metrics": {"y": 1.0}, "status": "observed", "observed_at": 1},
            {"trial_id": "b", "metrics": {"y": 4.0}, "status": "observed", "observed_at": 2},
            {"trial_id": "c", "metrics": {"y": 2.0}, "status": "observed", "observed_at": 3},
        ],
    }
    conv = _compute_convergence(state)
    assert [p["best"] for p in conv["points"]] == [1.0, 4.0, 4.0]


def test_compute_convergence_none_for_multi_objective():
    state = {"shelf": {"objectives": [{"metric": "y1", "minimize": True}, {"metric": "y2", "minimize": False}]}, "trials": []}
    assert _compute_convergence(state) is None


def test_run_kind_distinguishes_agent_vs_lenz_only_runs():
    assert _run_kind(meta=None, has_trace=False) == "lenz"
    assert _run_kind(meta=None, has_trace=True) == "sara"
    assert _run_kind(meta={"provider": "anthropic"}, has_trace=False) == "sara"
    assert _run_kind(meta={"kind": "sara"}, has_trace=False) == "sara"
    assert _run_kind(meta={"kind": "lenz", "policy": "vanilla"}, has_trace=False) == "lenz"


def test_duration_seconds_from_meta_timestamps():
    meta = {"started_at": "2026-01-01T00:00:00+00:00", "ended_at": "2026-01-01T00:05:00+00:00"}
    assert _duration_seconds({}, meta, []) == 300.0


def test_duration_seconds_from_trace_when_no_meta():
    trace = [{"ts": 1000.0}, {"ts": 1010.0}, {"ts": 1042.5}]
    assert _duration_seconds({}, None, trace) == 42.5


def test_duration_seconds_none_when_no_signal():
    assert _duration_seconds({"trials": []}, None, []) is None


def test_find_runs_includes_meta_and_sorts_by_recency(tmp_path):
    (tmp_path / "older").mkdir()
    _write_state(tmp_path / "older" / "state.json", [{"metric": "y", "minimize": True}], [])
    (tmp_path / "older" / "run_meta.json").write_text(
        json.dumps({"provider": "anthropic", "model": "claude-x", "status": "completed", "started_at": "2026-01-01T00:00:00+00:00"})
    )
    (tmp_path / "newer").mkdir()
    _write_state(tmp_path / "newer" / "state.json", [{"metric": "y", "minimize": True}], [])
    (tmp_path / "newer" / "run_meta.json").write_text(
        json.dumps({"provider": "openai", "model": "gpt-x", "status": "running", "started_at": "2026-06-01T00:00:00+00:00"})
    )

    runs = find_runs(tmp_path)
    assert [r["name"] for r in runs] == ["newer", "older"]  # most recent first
    assert runs[0]["provider"] == "openai" and runs[0]["model"] == "gpt-x" and runs[0]["status"] == "running"
    assert runs[0]["run_kind"] == "sara"


def test_run_detail_includes_meta_and_duration(tmp_path):
    (tmp_path / "run1").mkdir()
    _write_state(tmp_path / "run1" / "state.json", [{"metric": "y", "minimize": True}], [])
    (tmp_path / "run1" / "run_meta.json").write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-x",
                "status": "completed",
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:01:40+00:00",
            }
        )
    )
    detail = run_detail(tmp_path, "run1")
    assert detail["meta"]["model"] == "claude-x"
    assert detail["run_kind"] == "sara"
    assert detail["duration_seconds"] == 100.0


def test_run_detail_lenz_only_run_has_no_meta(tmp_path):
    (tmp_path / "run1").mkdir()
    _write_state(tmp_path / "run1" / "state.json", [{"metric": "y", "minimize": True}], [])
    detail = run_detail(tmp_path, "run1")
    assert detail["meta"] is None
    assert detail["run_kind"] == "lenz"


def test_run_detail_trace_without_meta_is_sara(tmp_path):
    """Blind sara runs used to write trace.jsonl but not run_meta.json.
    Missing meta must not be treated as a lenz-only (vanilla BO) run."""
    run_dir = tmp_path / "sara-lenz" / "sandbox_abc"
    run_dir.mkdir(parents=True)
    _write_state(run_dir / "state.json", [{"metric": "y", "minimize": True}], [])
    (run_dir / "trace.jsonl").write_text('{"role": "user", "content": "go"}\n')
    detail = run_detail(tmp_path, "sara-lenz/sandbox_abc")
    assert detail["meta"] is None
    assert detail["run_kind"] == "sara"


def test_run_detail_explicit_lenz_kind_in_meta(tmp_path):
    (tmp_path / "vanilla").mkdir()
    _write_state(tmp_path / "vanilla" / "state.json", [{"metric": "y", "minimize": True}], [])
    (tmp_path / "vanilla" / "run_meta.json").write_text(json.dumps({"kind": "lenz", "policy": "vanilla"}))
    detail = run_detail(tmp_path, "vanilla")
    assert detail["run_kind"] == "lenz"


def test_run_detail_started_at_falls_back_to_trial_times(tmp_path):
    (tmp_path / "run1").mkdir()
    trials = [
        {"trial_id": "a", "config": {"x": 0.1}, "metrics": {"y": 1.0}, "status": "observed", "created_at": 1_700_000_000.0},
    ]
    _write_state(tmp_path / "run1" / "state.json", [{"metric": "y", "minimize": True}], trials)
    detail = run_detail(tmp_path, "run1")
    assert detail["started_at"] is not None
    assert "2023" in detail["started_at"] or "2024" in detail["started_at"]


def test_delete_run_removes_directory(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_state(run_dir / "state.json", [{"metric": "y", "minimize": True}], [])
    (run_dir / "trace.jsonl").write_text('{"role": "user", "content": "hi"}\n')

    assert delete_run(tmp_path, "run1") is True
    assert not run_dir.exists()


def test_delete_run_nested_path(tmp_path):
    run_dir = tmp_path / "group" / "run1"
    run_dir.mkdir(parents=True)
    _write_state(run_dir / "state.json", [{"metric": "y", "minimize": True}], [])

    assert delete_run(tmp_path, "group/run1") is True
    assert not run_dir.exists()
    assert (tmp_path / "group").exists()  # only the run's own directory is removed


def test_delete_run_missing_returns_false(tmp_path):
    assert delete_run(tmp_path, "does-not-exist") is False


def test_delete_run_blocks_path_traversal(tmp_path):
    outside = tmp_path.parent / "should-not-be-touched"
    outside.mkdir(exist_ok=True)
    (outside / "state.json").write_text("{}")
    try:
        assert delete_run(tmp_path, "../should-not-be-touched") is False
        assert outside.exists()
    finally:
        import shutil

        shutil.rmtree(outside, ignore_errors=True)


def test_delete_run_cannot_delete_root_itself(tmp_path):
    (tmp_path / "state.json").write_text("{}")
    assert delete_run(tmp_path, ".") is False
    assert tmp_path.exists()


def test_kernel_generations_extracts_evolve_events():
    state = {
        "events": [
            {"command": "submit", "trial_id": "a"},
            {"command": "evolve-kernels", "generation": 1, "population": ["SE", "PER"], "best": "SE"},
            {"command": "evolve-kernels", "status": "skipped"},  # no generation/population -> excluded
            {"command": "evolve-kernels", "generation": 2, "population": ["SE", "LIN"], "best": "LIN"},
        ]
    }
    gens = _kernel_generations(state)
    assert [g["generation"] for g in gens] == [1, 2]


def test_reconfigurations_filters_to_mutating_backend_commands():
    state = {
        "events": [
            {"command": "create"},
            {"command": "suggest", "q": 1},
            {"command": "submit", "trial_id": "a"},
            {"command": "set-acqf", "acqf": "ucb"},
            {"command": "set-bounds", "bounds": {"x": [0.0, 0.5]}},
            {"command": "evolve-kernels", "generation": 1},
        ]
    }
    reconfigs = _reconfigurations(state)
    assert [r["command"] for r in reconfigs] == ["set-acqf", "set-bounds"]


def test_classify_lenz_calls_splits_suggest_by_flag():
    assert _classify_lenz_calls("lenz suggest --state ./state.json") == ["suggest"]
    assert _classify_lenz_calls("lenz suggest --state ./state.json --bounds '{}'") == ["suggest_bounded"]
    assert _classify_lenz_calls("lenz suggest --state ./state.json --around abc123") == ["suggest_around"]


def test_classify_lenz_calls_scopes_flags_to_the_right_chained_invocation():
    # the --bounds flag belongs to the FIRST suggest call; the second is plain
    cmd = "lenz suggest --state s --bounds '{}' && lenz suggest --state s"
    assert _classify_lenz_calls(cmd) == ["suggest_bounded", "suggest"]


def test_classify_lenz_calls_other_subcommands_pass_through():
    cmd = "lenz status --state s && lenz diagnostics --state s"
    assert _classify_lenz_calls(cmd) == ["status", "diagnostics"]


def test_tool_use_events_tags_call_type_and_progress():
    state = {
        "trials": [
            {"status": "observed", "observed_at": 100.0},
            {"status": "observed", "observed_at": 200.0},
        ]
    }
    trace = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "ts": 100.0,  # exactly at the first observed trial -> 1/2 progress
            "tool_calls": [{"name": "bash", "arguments": {"cmd": "lenz status --state ./state.json"}}],
        },
        {
            "role": "assistant",
            "ts": 250.0,  # after both trials -> full progress
            "tool_calls": [{"name": "bash", "arguments": {"cmd": "lenz suggest --state ./state.json --around x"}}],
        },
        {
            "role": "assistant",
            "ts": 260.0,
            "tool_calls": [{"name": "read", "arguments": {"path": "context.md"}}],  # not bash -> ignored
        },
    ]
    events = _tool_use_events(state, trace)
    assert events == [
        {"call_type": "status", "progress": 0.5},
        {"call_type": "suggest_around", "progress": 1.0},
    ]


def test_tool_use_events_none_without_trace_or_observed_trials():
    assert _tool_use_events({"trials": []}, [{"role": "assistant", "tool_calls": []}]) is None
    assert _tool_use_events({"trials": [{"status": "observed", "observed_at": 1.0}]}, []) is None
