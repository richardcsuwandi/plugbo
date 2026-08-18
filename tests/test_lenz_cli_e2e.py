"""End-to-end sanity check: script the lenz CLI (subprocess, exactly as an
agent would call it) through a full create -> suggest/submit -> incumbent
loop on the Branin function, and confirm the surrogate-guided phase beats
the pure-Sobol warm-up.
"""

import json
import math
import subprocess
import sys

import pytest

BRANIN_MIN = 0.397887


def branin(x1: float, x2: float) -> float:
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi, 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def _run(state_path, command, *args):
    out = subprocess.run(
        [sys.executable, "-m", "lenz.cli", command, "--state", str(state_path), *args],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["ok"], payload
    return payload["result"]


@pytest.mark.slow
def test_branin_regret_decreases(tmp_path):
    state = tmp_path / "state.json"
    _run(
        state,
        "create",
        "--space",
        json.dumps(
            {
                "x1": {"kind": "range", "lower": -5.0, "upper": 10.0},
                "x2": {"kind": "range", "lower": 0.0, "upper": 15.0},
            }
        ),
        "--objectives",
        json.dumps({"y": "minimize"}),
        "--acqf",
        "noisy_logei",
    )

    warmup_best = float("inf")
    budget = 20
    for i in range(budget):
        sug = _run(state, "suggest")
        cfg = sug[0]["config"]
        y = branin(cfg["x1"], cfg["x2"])
        _run(state, "submit", "--config", json.dumps(cfg), "--metrics", json.dumps({"y": y}))
        if i == 2:  # after the Sobol warm-up (threshold = d+1 = 3)
            inc = _run(state, "incumbent")
            warmup_best = inc["metrics"]["y"]

    final = _run(state, "incumbent")
    final_regret = final["metrics"]["y"] - BRANIN_MIN
    warmup_regret = warmup_best - BRANIN_MIN

    assert final_regret < warmup_regret
    assert final_regret < 2.0
