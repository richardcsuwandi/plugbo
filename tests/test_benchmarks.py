"""Correctness of the canonical benchmark math, the obfuscation transform
(rename / unit-hypercube / shift), and the generated blind-test sandbox --
including that no identifying string ever appears in anything the agent's
tools can see.
"""

import json
import subprocess
import sys

import pytest

from benchmarks.functions import get_spec
from benchmarks.obfuscate import ObfuscatedBenchmark, build_obfuscated
from benchmarks.sandbox import build_sandbox

IDENTIFYING_STRINGS = ["branin", "ackley", "hartmann"]


@pytest.mark.parametrize(
    "name,x_opt",
    [
        ("branin", [3.14159265, 2.275]),
        ("branin", [9.42478, 2.475]),
        ("branin", [-3.14159265, 12.275]),
        ("hartmann6", [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
        ("ackley10", [0.0] * 10),
        ("rosenbrock6", [1.0] * 6),
        ("rastrigin10", [0.0] * 10),
        ("levy6", [1.0] * 6),
        ("griewank10", [0.0] * 10),
        ("michalewicz2", [2.202906, 1.570796]),
        ("michalewicz10", [2.202905, 1.570797, 1.284992, 1.923058, 1.72047, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]),
        ("styblinski_tang6", [-2.903534027771177] * 6),
        ("shekel", [4.000747, 3.999509, 4.000747, 3.999509]),
        ("six_hump_camel", [0.089842, -0.712656]),
        ("constrained_hartmann6", [0.312390, 0.281858, 0.480840, 0.318579, 0.332827, 0.626447]),
    ],
)
def test_known_optimum_matches_spec(name, x_opt):
    spec = get_spec(name)
    assert spec.fn(x_opt) == pytest.approx(spec.f_opt, abs=1e-3)


def test_new_benchmarks_registered_and_in_bounds():
    for name in [
        "rosenbrock6",
        "rosenbrock10",
        "rastrigin6",
        "rastrigin10",
        "levy6",
        "levy10",
        "griewank10",
        "michalewicz2",
        "michalewicz5",
        "michalewicz10",
        "styblinski_tang6",
        "styblinski_tang10",
        "shekel",
        "six_hump_camel",
        "constrained_hartmann6",
        "bolt_lora",
    ]:
        spec = get_spec(name)
        assert spec.dim == len(spec.bounds)
        assert spec.name == name


def test_constrained_hartmann6_optimum_is_feasible_and_shifted_off_unconstrained_optimum():
    spec = get_spec("constrained_hartmann6")
    unconstrained_x_opt = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
    # the unconstrained Hartmann6 optimum must NOT satisfy the constraint --
    # otherwise "constrained" Hartmann6 wouldn't force any real search
    assert spec.constraint_fn(unconstrained_x_opt) > spec.constraint_upper
    # but the constrained optimum we report must itself be feasible
    x_opt = [0.312390, 0.281858, 0.480840, 0.318579, 0.332827, 0.626447]
    assert spec.constraint_fn(x_opt) <= spec.constraint_upper + 1e-6
    # and strictly worse (higher, since we minimize) than the unconstrained optimum
    assert spec.fn(x_opt) > spec.fn(unconstrained_x_opt)


def test_gp_sample_is_deterministic_in_seed_and_varies_across_seeds():
    from benchmarks.functions import get_spec as _get_spec

    a1 = _get_spec("gp_sample4", seed=7)
    a2 = _get_spec("gp_sample4", seed=7)
    b = _get_spec("gp_sample4", seed=8)
    probe = [0.2, 0.4, 0.6, 0.8]
    assert a1.fn(probe) == pytest.approx(a2.fn(probe))
    assert a1.fn(probe) != pytest.approx(b.fn(probe))
    assert a1.dim == 4
    assert a1.bounds == [(0.0, 1.0)] * 4


def test_gp_sample_requires_seed():
    with pytest.raises(ValueError):
        get_spec("gp_sample4")


def test_gp_sample_f_opt_is_near_the_true_minimum():
    spec = get_spec("gp_sample3", seed=99)
    # the numerically-estimated f_opt should not be beaten by more than a tiny
    # margin by a coarse independent grid search -- if it is, n_restarts is too small
    import itertools

    grid = [i / 10.0 for i in range(11)]
    worst_gap = min(spec.fn(list(pt)) for pt in itertools.product(grid, repeat=3)) - spec.f_opt
    assert worst_gap > -1e-2


def test_obfuscate_recovers_true_optimum_through_shift():
    spec = get_spec("hartmann6")
    x_opt = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
    for seed in range(5):
        ob = build_obfuscated("hartmann6", seed=seed)
        config = ob.true_to_unit(x_opt)
        assert all(0.0 <= v <= 1.0 for v in config.values())
        y = ob.evaluate(config)
        assert y == pytest.approx(spec.f_opt, abs=1e-3)


def test_obfuscate_renames_and_shifts_nontrivially():
    spec = get_spec("ackley10")
    ob = build_obfuscated("ackley10", seed=42)
    assert set(ob.param_names) != set(f"x{i}" for i in range(spec.dim))
    assert len(ob.param_names) == spec.dim == len(set(ob.param_names))
    assert any(abs(s) > 1e-6 for s in ob.shift_frac)
    assert all(-0.25 <= s <= 0.25 for s in ob.shift_frac)


def test_obfuscate_unit_space_is_generic():
    ob = build_obfuscated("branin", seed=1)
    space = ob.unit_space_json()
    assert set(space) == set(ob.param_names)
    for d in space.values():
        assert d == {"kind": "range", "lower": 0.0, "upper": 1.0}


def test_secret_roundtrip():
    ob = build_obfuscated("ackley20", seed=7)
    restored = ObfuscatedBenchmark.from_secret(ob.to_secret())
    assert restored.param_names == ob.param_names
    assert restored.shift_frac == pytest.approx(ob.shift_frac)
    assert restored.spec.name == ob.spec.name


@pytest.mark.parametrize("benchmark_name", ["branin", "hartmann6", "ackley10"])
def test_sandbox_hides_identity_and_oracle_matches_reference(tmp_path, benchmark_name):
    built = build_sandbox(benchmark_name, root=tmp_path, seed=123)
    sandbox = built["sandbox"]

    # nothing visible to the agent (context.md, token, sandbox dir name) names the benchmark
    context = (sandbox / "context.md").read_text()
    dirname = sandbox.name
    token = built["token"]
    for s in IDENTIFYING_STRINGS:
        assert s not in context.lower()
        assert s not in dirname.lower()
    assert token in dirname

    # the oracle's own source (reachable via the symlink) doesn't name the benchmark either
    oracle_source = (sandbox / "oracle").resolve().read_text()
    for s in IDENTIFYING_STRINGS:
        assert s not in oracle_source.lower()

    # the secret lives outside the sandbox, not reachable via any relative path from inside it
    assert not (sandbox / "..").resolve().joinpath(f"_answers/{token}.json").is_relative_to(sandbox)

    # executing the oracle reproduces ObfuscatedBenchmark.evaluate() exactly
    secret = json.loads(built["secret_path"].read_text())
    ob = ObfuscatedBenchmark.from_secret(secret)
    config = {name: 0.3 + 0.1 * i for i, name in enumerate(ob.param_names)}
    expected_y = ob.evaluate(config)

    out = subprocess.run(
        [sys.executable, str(sandbox / "oracle"), json.dumps(config)],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    got_y = json.loads(out.stdout)["y"]
    assert got_y == pytest.approx(expected_y, rel=1e-9)


def test_sandbox_constrained_oracle_reports_y_and_c(tmp_path):
    built = build_sandbox("constrained_hartmann6", root=tmp_path, seed=11)
    sandbox = built["sandbox"]
    assert built["constraints"] == [{"metric": "c", "upper": 0.0}]

    context = (sandbox / "context.md").read_text()
    assert '"c"' in context
    for s in IDENTIFYING_STRINGS:
        assert s not in context.lower()  # still blind: constraint disclosure doesn't leak identity

    ob = ObfuscatedBenchmark.from_secret(json.loads(built["secret_path"].read_text()))
    config = {name: 0.4 for name in ob.param_names}
    expected = ob.evaluate_metrics(config)

    out = subprocess.run(
        [sys.executable, str(sandbox / "oracle"), json.dumps(config)], cwd=sandbox, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["y"] == pytest.approx(expected["y"], rel=1e-9)
    assert got["c"] == pytest.approx(expected["c"], rel=1e-9)


def test_sandbox_gp_sample_oracle_matches_reference(tmp_path):
    built = build_sandbox("gp_sample5", root=tmp_path, seed=17)
    sandbox = built["sandbox"]
    ob = ObfuscatedBenchmark.from_secret(json.loads(built["secret_path"].read_text()))
    assert ob.spec.dim == 5

    config = {name: 0.1 + 0.15 * i for i, name in enumerate(ob.param_names)}
    expected_y = ob.evaluate(config)
    out = subprocess.run(
        [sys.executable, str(sandbox / "oracle"), json.dumps(config)], cwd=sandbox, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    got_y = json.loads(out.stdout)["y"]
    assert got_y == pytest.approx(expected_y, rel=1e-6)


@pytest.mark.parametrize("shift", [False, True])
def test_sandbox_reveal_mode_states_identity_and_true_bounds(tmp_path, shift):
    built = build_sandbox("hartmann6", root=tmp_path, seed=21, reveal=True, shift=shift)
    sandbox = built["sandbox"]

    context = (sandbox / "context.md").read_text()
    assert "hartmann6" in context.lower()

    ob = ObfuscatedBenchmark.from_secret(json.loads(built["secret_path"].read_text()))
    assert ob.param_names == ["x1", "x2", "x3", "x4", "x5", "x6"]
    space = ob.unit_space_json()
    assert space["x1"] == {"kind": "range", "lower": 0.0, "upper": 1.0}  # hartmann6's true bounds happen to be [0,1]

    x_opt = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
    config = dict(zip(ob.param_names, x_opt))
    out = subprocess.run(
        [sys.executable, str(sandbox / "oracle"), json.dumps(config)], cwd=sandbox, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    y = json.loads(out.stdout)["y"]
    if shift:
        assert y > ob.spec.f_opt + 0.1  # textbook optimum no longer solves it
    else:
        assert y == pytest.approx(ob.spec.f_opt, abs=1e-3)  # pure recall: submitting it verbatim wins


def test_reveal_mode_exposes_true_bounds_for_a_non_unit_cube_benchmark(tmp_path):
    # Ackley's true domain is [-32.768, 32.768], unlike Hartmann6's coincidental [0, 1] --
    # this is the case that actually exercises reveal_bounds's unit conversion.
    built = build_sandbox("ackley10", root=tmp_path, seed=3, reveal=True, shift=False)
    sandbox = built["sandbox"]
    ob = ObfuscatedBenchmark.from_secret(json.loads(built["secret_path"].read_text()))
    space = ob.unit_space_json()
    assert space["x1"] == {"kind": "range", "lower": -32.768, "upper": 32.768}

    config = {name: 0.0 for name in ob.param_names}  # Ackley's known optimum, in true units
    out = subprocess.run(
        [sys.executable, str(sandbox / "oracle"), json.dumps(config)], cwd=sandbox, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    y = json.loads(out.stdout)["y"]
    assert y == pytest.approx(ob.spec.f_opt, abs=1e-6)


def test_run_noblind_test_writes_meta_and_one_shot_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks import run_noblind_test as m

    sandbox = tmp_path / "sandbox_xyz"
    sandbox.mkdir()
    (sandbox / "context.md").write_text("this is hartmann6, revealed")

    monkeypatch.setattr(
        m,
        "build_sandbox",
        lambda *a, **k: {"sandbox": sandbox, "secret_path": tmp_path / "secret.json", "constraints": None},
    )
    (tmp_path / "secret.json").write_text(
        json.dumps(
            {
                "benchmark": "hartmann6",
                "param_names": [f"x{i + 1}" for i in range(6)],
                "shift_frac": [0.0] * 6,
                "seed": 1,
                "reveal_bounds": True,
            }
        )
    )
    monkeypatch.setattr(m, "create_and_warmup", lambda *a, **k: 0)
    monkeypatch.setattr(m, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(m, "run_campaign", lambda **k: SimpleNamespace(final_message="done", n_steps=1, usage_total=None))
    monkeypatch.setattr(
        m,
        "score_sandbox",
        lambda *a, **k: {
            "benchmark": "hartmann6",
            "true_f_opt": -3.32237,
            "n_observed": 1,
            "best_regret": 0.0005,
            "trials": [{"config": {}, "y": -3.3218700, "regret": 0.0005}],
            "sandbox": str(sandbox),
        },
    )

    result = m.run_noblind_test(
        benchmark_name="hartmann6",
        provider="openai-compatible",
        model="Qwen-Ambassador/Qwen3.8-Max",
        budget=5,
        root=tmp_path,
        base_url="https://example.test",
    )

    meta = json.loads((sandbox / "run_meta.json").read_text())
    assert meta["kind"] == "sara-noblind"
    assert meta["reveal"] is True
    assert meta["status"] == "completed"
    assert result["one_shot_success"] is True
    assert result["first_trial_regret"] == pytest.approx(0.0005)


def test_sandbox_read_tool_cannot_escape_to_the_secret(tmp_path):
    from sara.tools import build_tools

    built = build_sandbox("branin", root=tmp_path, seed=5)
    sandbox = built["sandbox"]
    _, handlers = build_tools(sandbox)

    # the oracle symlink resolves outside the sandbox, so the `read` tool -- unlike `bash` --
    # refuses it, matching the system prompt's "don't read the implementation" rule technically
    assert "outside the sandbox" in handlers["read"]("oracle")
    assert "hello" not in handlers["read"]("../_answers").lower()  # sanity: no accidental traversal


def test_run_blind_test_writes_sara_run_meta(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks import run_blind_test as m

    sandbox = tmp_path / "sandbox_abc"
    sandbox.mkdir()
    (sandbox / "context.md").write_text("minimize y")

    monkeypatch.setattr(
        m,
        "build_sandbox",
        lambda *a, **k: {"sandbox": sandbox, "secret_path": tmp_path / "secret.json", "constraints": None},
    )
    (tmp_path / "secret.json").write_text(
        json.dumps({"benchmark": "hartmann6", "param_names": list("abcdef"), "shift_frac": [0.0] * 6})
    )
    monkeypatch.setattr(m, "create_and_warmup", lambda *a, **k: 0)
    monkeypatch.setattr(m, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(
        m,
        "run_campaign",
        lambda **k: SimpleNamespace(final_message="done", n_steps=3, usage_total=None),
    )
    monkeypatch.setattr(
        m,
        "score_sandbox",
        lambda *a, **k: {
            "benchmark": "hartmann6",
            "true_f_opt": 0.0,
            "n_observed": 0,
            "best_regret": 0.0,
            "trials": [],
            "sandbox": str(sandbox),
        },
    )

    m.run_blind_test(
        benchmark_name="hartmann6",
        provider="openai-compatible",
        model="Qwen-Ambassador/Qwen3.8-Max",
        budget=50,
        root=tmp_path,
        base_url="https://example.test",
    )

    meta = json.loads((sandbox / "run_meta.json").read_text())
    assert meta["kind"] == "sara"
    assert meta["status"] == "completed"
    assert meta["provider"] == "openai-compatible"
    assert meta["model"] == "Qwen-Ambassador/Qwen3.8-Max"
    assert meta["budget"] == 50
    assert meta["n_steps"] == 3
    assert meta["started_at"] and meta["ended_at"]


def test_plot_compare_uses_newest_sandbox(tmp_path):
    import os
    import time

    from benchmarks.plot_compare import regret_trace
    from benchmarks.obfuscate import build_obfuscated

    cond = tmp_path / "vanilla"
    answers = cond / "_answers"
    answers.mkdir(parents=True)
    ob = build_obfuscated("branin", seed=1)
    secret = ob.to_secret()

    def write_run(token, y0, mtime):
        sandbox = cond / f"sandbox_{token}"
        sandbox.mkdir()
        (answers / f"{token}.json").write_text(json.dumps(secret))
        names = ob.param_names
        state = {
            "space": {n: {"kind": "range", "lower": 0.0, "upper": 1.0} for n in names},
            "shelf": {"objectives": [{"metric": "y", "minimize": True}]},
            "trials": [
                {"status": "observed", "config": {n: 0.5 for n in names}, "metrics": {"y": y0}},
            ],
        }
        path = sandbox / "state.json"
        path.write_text(json.dumps(state))
        os.utime(path, (mtime, mtime))

    now = time.time()
    write_run("aaaa", y0=10.0, mtime=now)
    write_run("zzzz", y0=1.0, mtime=now + 10)  # newer; name-sort would have picked aaaa

    trace = regret_trace(cond)
    assert trace is not None
    assert abs(trace[0] - (1.0 - ob.spec.f_opt)) < 1e-6


def test_plot_compare_renders_single_eval_trace():
    import re

    from benchmarks.plot_compare import _chart_svg

    svg = _chart_svg({"blind": [3.0, 2.0, 1.0], "noblind": [0.0]})
    polys = re.findall(r'<polyline points="([^"]+)"', svg)
    assert len(polys) == 2
    assert len(polys[1].split()) >= 2  # 1-eval run must pad to a visible segment
    assert "1 eval" in svg


def test_pick_legend_corner_prefers_empty_bottom_left_for_high_start_curves():
    from benchmarks.plot_compare import pick_legend_corner

    # High y (near the top of the plot) across most of the x range, like Ackley.
    x0, y1, x1, y0 = 60, 16, 800, 404
    w, h = 306, 102
    points = [(x, 30) for x in range(70, 790, 8)]
    assert pick_legend_corner(points, x0, y1, x1, y0, w, h) == "bl"


def test_pick_legend_corner_avoids_crowded_bottom_right():
    from benchmarks.plot_compare import pick_legend_corner

    x0, y1, x1, y0 = 60, 16, 800, 404
    w, h = 306, 102
    points = [(x, 390) for x in range(500, 790, 4)]
    assert pick_legend_corner(points, x0, y1, x1, y0, w, h) == "tr"


def test_pick_legend_corner_prefers_top_right_when_top_is_empty():
    from benchmarks.plot_compare import pick_legend_corner

    x0, y1, x1, y0 = 60, 16, 800, 404
    w, h = 306, 102
    points = [(x, 390) for x in range(70, 790, 8)]
    assert pick_legend_corner(points, x0, y1, x1, y0, w, h) == "tr"


def test_warmup_n_defaults():
    from benchmarks.lenz_loop import warmup_n

    assert warmup_n(6, seed=None, warmup=None) == 0
    assert warmup_n(6, seed=42, warmup=None) == 7
    assert warmup_n(6, seed=42, warmup=0) == 0
    assert warmup_n(2, seed=42, warmup=None) == 3
