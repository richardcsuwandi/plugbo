"""Shared warm-start behavior for run_noblind_test.py's --no-lenz path,
exercised here because bolt_lora-compare's sara-only leg uses this module.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks import run_noblind_test as m


def _fake_score_sandbox(sandbox):
    return lambda *a, **k: {
        "benchmark": "hartmann6",
        "true_f_opt": 0.0,
        "n_observed": 0,
        "best_regret": float("inf"),
        "trials": [],
        "sandbox": str(sandbox),
    }


def test_run_noblind_test_no_lenz_uses_shared_default_warmup(tmp_path, monkeypatch):
    import benchmarks.sara_only as sara_only_mod

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
    monkeypatch.setattr(m, "score_sandbox", _fake_score_sandbox(sandbox))

    m.run_noblind_test("hartmann6", provider="openai", model="x", budget=10, root=tmp_path, seed=1, no_lenz=True)
    assert warm_calls == [7]
    assert install_calls[-1]["preserve_trials"] is True
    assert captured.get("block_lenz") is True
    assert "lenz" not in captured["user_prompt"].lower()
    assert "7 evaluation(s) already ran" in captured["user_prompt"]
    meta = json.loads((sandbox / "run_meta.json").read_text())
    assert meta["kind"] == "sara-only"
    assert meta["warmup"] == 7
    assert (sandbox / ".oracle_impl").is_file()


def test_run_noblind_test_no_lenz_explicit_warmup_matches_lenz_siblings(tmp_path, monkeypatch):
    import benchmarks.sara_only as sara_only_mod

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
    monkeypatch.setattr(m, "score_sandbox", _fake_score_sandbox(sandbox))

    m.run_noblind_test(
        "hartmann6", provider="openai", model="x", budget=10, root=tmp_path, seed=1, no_lenz=True, warmup=3
    )
    assert warm_calls == [3]
    assert install_calls[-1]["preserve_trials"] is True
    assert "lenz" not in captured["user_prompt"].lower()
    assert "3 evaluation(s) already ran" in captured["user_prompt"]
    meta = json.loads((sandbox / "run_meta.json").read_text())
    assert meta["warmup"] == 3
