"""Thin subprocess wrappers around the `lenz` CLI, shared by the scripted
baseline and the sara-driven blind test (so both can own the same Sobol
warm-start before their policies diverge).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def lenz(state_path: Path, command: str, *args: str) -> dict:
    # `--state` is a per-subcommand flag, not a global one -- must come after
    # the subcommand (`lenz create --state ./state.json ...`), per lenz/cli.py.
    out = subprocess.run(
        ["lenz", command, "--state", str(state_path), *args], capture_output=True, text=True, check=True
    )
    payload = json.loads(out.stdout)
    if not payload["ok"]:
        raise RuntimeError(payload["error"])
    return payload["result"]


def evaluate(oracle_path: Path, config: dict) -> dict:
    out = subprocess.run(
        [str(oracle_path), json.dumps(config)], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def warmup_n(dim: int, seed: int | None, warmup: int | None) -> int:
    """Shared Sobol warm-start length. Default: d+1 (lenz's own GP warmup)
    when a seed is set, else 0 (leave the opening to the method).
    """
    if warmup is not None:
        return max(0, warmup)
    if seed is None:
        return 0
    return max(2, dim + 1)


def create_and_warmup(
    sandbox: Path,
    space: dict,
    create_args: list[str],
    n_warmup: int,
    budget: int,
) -> int:
    """Creates `sandbox/state.json` and evaluates `n_warmup` Sobol points.
    Returns how many evaluations were recorded (clamped to `budget`).
    """
    state_path = sandbox / "state.json"
    oracle_path = sandbox / "oracle"
    n_warmup = min(n_warmup, budget)
    lenz(
        state_path,
        "create",
        "--space",
        json.dumps(space),
        "--objectives",
        json.dumps({"y": "minimize"}),
        *create_args,
    )
    for i in range(1, n_warmup + 1):
        config = lenz(state_path, "suggest")[0]["config"]
        metrics = evaluate(oracle_path, config)
        lenz(state_path, "submit", "--config", json.dumps(config), "--metrics", json.dumps(metrics))
        # `incumbent`'s metrics are None when the study is constrained and no
        # submitted trial has been feasible yet (common during warm-up, since
        # Sobol samples the whole box, not just the feasible region).
        incumbent_metrics = lenz(state_path, "incumbent")["metrics"]
        best = f"{incumbent_metrics['y']:.4f}" if incumbent_metrics else "none yet (infeasible)"
        print(f"eval {i:3d}/{budget}  warmup  y={metrics['y']:.4f}  best-so-far={best}")
    return n_warmup
