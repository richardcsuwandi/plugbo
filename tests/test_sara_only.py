"""Sara-only (no lenz): tool block, oracle wrapper, prompt hygiene."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from sara.cli import _system_prompt_sara_only, _user_prompt_sara_only
from sara.tools import build_tools
from benchmarks.sara_only import install_sara_only
from benchmarks.sandbox import build_sandbox
from benchmarks.run_blind_test import score_sandbox


def test_sara_only_prompts_never_mention_lenz():
    sys_p = _system_prompt_sara_only().lower()
    user = _user_prompt_sara_only("minimize y over x in [0,1]", "./oracle", 10).lower()
    assert "lenz" not in sys_p
    assert "lenz" not in user
    assert "gaussian process" not in sys_p
    assert "botorch" not in sys_p


def test_block_lenz_rejects_cli_and_import(tmp_path):
    _, handlers = build_tools(tmp_path, block_lenz=True)
    for cmd in ("lenz status", "lenz suggest --state ./state.json", "python3 -m lenz", "python -c 'import lenz'"):
        out = handlers["bash"](cmd)
        assert "not available" in out, cmd
        assert "PWNED" not in out


def test_block_lenz_shadows_split_import(tmp_path):
    """A command that never spells the backend name still cannot import it."""
    _, handlers = build_tools(tmp_path, block_lenz=True)
    out = handlers["bash"]("python3 -c \"import importlib; importlib.import_module('le'+'nz'); print('PWNED')\"")
    assert "PWNED" not in out
    assert "no external optimizer" in out or "Error" in out or "error" in out.lower()


def test_block_lenz_strips_lenz_from_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "lenz"
    fake.write_text("#!/bin/sh\necho PWNED\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    _, handlers = build_tools(tmp_path, block_lenz=True)
    path_out = handlers["bash"]("printf '%s' \"$PATH\"")
    assert str(bin_dir) not in path_out
    _, open_handlers = build_tools(tmp_path, block_lenz=False)
    assert "hello" in open_handlers["bash"]("echo hello")


def test_unblocked_bash_still_runs(tmp_path):
    _, handlers = build_tools(tmp_path, block_lenz=False)
    assert "hello" in handlers["bash"]("echo hello")


def test_wrapper_records_trials_and_enforces_budget(tmp_path):
    oracle = tmp_path / "oracle"
    oracle.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({'y': 1.5}))\n")
    oracle.chmod(oracle.stat().st_mode | stat.S_IXUSR)
    install_sara_only(
        tmp_path,
        space={"x": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"}},
        objectives={"y": "minimize"},
        constraints=None,
        budget=1,
    )
    assert (tmp_path / ".oracle_impl").is_file()
    out = subprocess.run(
        [sys.executable, str(tmp_path / "oracle"), json.dumps({"x": 0.2})],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["y"] == pytest.approx(1.5)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["shelf"]["surrogate"] == "none"
    assert state["trials"][0]["config"]["x"] == pytest.approx(0.2)
    assert state["trials"][0]["status"] == "observed"

    denied = subprocess.run(
        [sys.executable, str(tmp_path / "oracle"), json.dumps({"x": 0.9})],
        capture_output=True,
        text=True,
    )
    assert denied.returncode != 0
    assert "budget" in denied.stdout.lower() + denied.stderr.lower()
    state2 = json.loads((tmp_path / "state.json").read_text())
    assert len(state2["trials"]) == 1


def test_wrapper_bakes_generation_interpreter_ignoring_broken_path_python3(tmp_path, monkeypatch):
    """Regression: `.oracle_impl` needs whatever deps the *harness's*
    interpreter has installed (e.g. `benchmarks`/`torch` for a surrogate
    oracle). A bare `#!/usr/bin/env python3` wrapper picks up whatever
    `python3` is first on the agent's own PATH instead, which broke
    bolt_lora's sara-only runs with `ModuleNotFoundError: No module named
    'benchmarks'`. The wrapper must bake in the generation-time interpreter
    (both its own shebang and the interpreter it uses to invoke IMPL) so
    the agent's PATH can't matter.
    """
    oracle = tmp_path / "oracle"
    oracle.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({'y': 2.5}))\n")
    oracle.chmod(oracle.stat().st_mode | stat.S_IXUSR)
    install_sara_only(
        tmp_path,
        space={"x": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"}},
        objectives={"y": "minimize"},
        constraints=None,
        budget=1,
    )

    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    broken = fake_bin / "python3"
    broken.write_text("#!/bin/sh\necho 'wrong interpreter' >&2\nexit 1\n")
    broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    # Exec `./oracle` directly (not `python3 ./oracle`) so the OS resolves
    # its shebang, exactly like the agent's `bash` tool running `./oracle`.
    out = subprocess.run(
        [str(tmp_path / "oracle"), json.dumps({"x": 0.3})],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["y"] == pytest.approx(2.5)


def test_install_sara_only_preserve_trials(tmp_path):
    oracle = tmp_path / "oracle"
    oracle.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps({'y': 1.0}))\n")
    oracle.chmod(oracle.stat().st_mode | stat.S_IXUSR)
    space = {"x": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"}}
    install_sara_only(tmp_path, space=space, objectives={"y": "minimize"}, constraints=None, budget=5)
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "space": space,
                "shelf": {"surrogate": "none"},
                "trials": [{"config": {"x": 0.1}, "metrics": {"y": 3.0}, "status": "observed"}],
                "events": [],
                "pending_x_gp": [],
            }
        )
    )

    install_sara_only(tmp_path, space=space, objectives={"y": "minimize"}, constraints=None, budget=5, preserve_trials=True)
    state = json.loads((tmp_path / "state.json").read_text())
    assert len(state["trials"]) == 1
    assert state["trials"][0]["config"]["x"] == pytest.approx(0.1)

    install_sara_only(tmp_path, space=space, objectives={"y": "minimize"}, constraints=None, budget=5)
    state2 = json.loads((tmp_path / "state.json").read_text())
    assert state2["trials"] == []


def test_wrapper_score_sandbox_roundtrip(tmp_path):
    built = build_sandbox("branin", root=tmp_path, seed=3)
    sandbox: Path = built["sandbox"]
    secret = json.loads(built["secret_path"].read_text())
    from benchmarks.obfuscate import ObfuscatedBenchmark

    ob = ObfuscatedBenchmark.from_secret(secret)
    space = ob.unit_space_json()
    install_sara_only(
        sandbox,
        space=space,
        objectives=ob.objectives_json(),
        constraints=None,
        budget=5,
    )
    ctx = (sandbox / "context.md").read_text().lower()
    assert "lenz" not in ctx
    cfg = json.dumps({name: 0.5 for name in space})
    out = subprocess.run([sys.executable, str(sandbox / "oracle"), cfg], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    y = json.loads(out.stdout)["y"]
    scored = score_sandbox(built, sandbox)
    assert scored["n_observed"] == 1
    assert scored["trials"][0]["y"] == pytest.approx(y)
    assert scored["best_regret"] < float("inf")


def test_run_blind_test_no_lenz_skips_warmup_and_blocks(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks import run_blind_test as m

    sandbox = tmp_path / "sandbox_nolenz"
    sandbox.mkdir()
    (sandbox / "context.md").write_text(
        "Use only `lenz` and `./oracle` to search; every evaluation must go through `./oracle`."
    )
    (sandbox / "oracle").write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps({'y': 0.0}))\n")

    captured = {}

    def fake_campaign(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(final_message="done", n_steps=1, usage_total=None)

    monkeypatch.setattr(
        m,
        "build_sandbox",
        lambda *a, **k: {"sandbox": sandbox, "secret_path": tmp_path / "secret.json", "constraints": None},
    )
    (tmp_path / "secret.json").write_text(
        json.dumps({"benchmark": "hartmann6", "param_names": list("abcdef"), "shift_frac": [0.0] * 6, "seed": 1})
    )

    def boom(*a, **k):
        raise AssertionError("create_and_warmup must not run in --no-lenz")

    monkeypatch.setattr(m, "create_and_warmup", boom)
    monkeypatch.setattr(m, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(m, "run_campaign", fake_campaign)
    monkeypatch.setattr(
        m,
        "score_sandbox",
        lambda *a, **k: {
            "benchmark": "hartmann6",
            "true_f_opt": 0.0,
            "n_observed": 0,
            "best_regret": float("inf"),
            "trials": [],
            "sandbox": str(sandbox),
        },
    )
    m.run_blind_test("hartmann6", provider="openai", model="x", budget=10, root=tmp_path, seed=1, no_lenz=True)
    assert captured.get("block_lenz") is True
    assert "lenz" not in captured["system_prompt"].lower()
    assert "lenz" not in captured["user_prompt"].lower()
    meta = json.loads((sandbox / "run_meta.json").read_text())
    assert meta["kind"] == "sara-only"
    assert meta["warmup"] == 0
    assert (sandbox / ".oracle_impl").is_file()


def test_run_blind_test_no_lenz_explicit_warmup_matches_lenz_siblings(tmp_path, monkeypatch):
    """Passing --warmup to a --no-lenz run should give it the same real
    Sobol warm-start its lenz-backed sibling conditions get -- generated by
    the harness (create_and_warmup), not the agent -- instead of always
    handing sara-only a free head start of zero pre-spent evaluations.
    """
    from types import SimpleNamespace

    import benchmarks.sara_only as sara_only_mod
    from benchmarks import run_blind_test as m

    sandbox = tmp_path / "sandbox_nolenz_warm"
    sandbox.mkdir()
    (sandbox / "context.md").write_text(
        "Use only `lenz` and `./oracle` to search; every evaluation must go through `./oracle`."
    )
    (sandbox / "oracle").write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps({'y': 0.0}))\n")

    captured = {}

    def fake_campaign(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(final_message="done", n_steps=1, usage_total=None)

    monkeypatch.setattr(
        m,
        "build_sandbox",
        lambda *a, **k: {"sandbox": sandbox, "secret_path": tmp_path / "secret.json", "constraints": None},
    )
    (tmp_path / "secret.json").write_text(
        json.dumps({"benchmark": "hartmann6", "param_names": list("abcdef"), "shift_frac": [0.0] * 6, "seed": 1})
    )

    warm_calls = []

    def fake_warmup(sandbox_arg, space, create_args, n_warmup, budget, objectives=None):
        warm_calls.append(n_warmup)
        return n_warmup

    install_calls = []
    real_install = sara_only_mod.install_sara_only

    def spy_install(*a, **k):
        install_calls.append(k)
        return real_install(*a, **k)

    monkeypatch.setattr(m, "create_and_warmup", fake_warmup)
    monkeypatch.setattr(sara_only_mod, "install_sara_only", spy_install)
    monkeypatch.setattr(m, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(m, "run_campaign", fake_campaign)
    monkeypatch.setattr(
        m,
        "score_sandbox",
        lambda *a, **k: {
            "benchmark": "hartmann6",
            "true_f_opt": 0.0,
            "n_observed": 0,
            "best_regret": float("inf"),
            "trials": [],
            "sandbox": str(sandbox),
        },
    )
    m.run_blind_test(
        "hartmann6", provider="openai", model="x", budget=10, root=tmp_path, seed=1, no_lenz=True, warmup=3
    )
    assert warm_calls == [3]
    assert install_calls[-1]["preserve_trials"] is True
    assert "lenz" not in captured["user_prompt"].lower()
    assert "3 evaluation(s) already ran" in captured["user_prompt"]
    meta = json.loads((sandbox / "run_meta.json").read_text())
    assert meta["warmup"] == 3
