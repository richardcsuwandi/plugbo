#!/usr/bin/env python3
"""Scripted lenz-only optimization loop on the Branin example -- no LLM
involved. Demonstrates that the BoTorch backend converges on its own; a fast
sanity check before wiring up an LLM provider.

Run from this directory: python3 run_manual.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "state.json"
BUDGET = 20
BRANIN_MIN = 0.397887


def lenz(*args: str) -> dict:
    out = subprocess.run(["lenz", "--state", str(STATE), *args], capture_output=True, text=True)
    payload = json.loads(out.stdout)
    if not payload["ok"]:
        raise RuntimeError(payload["error"])
    return payload["result"]


def evaluate(config: dict) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "eval.py"), json.dumps(config)], capture_output=True, text=True
    )
    return json.loads(out.stdout)


def main() -> None:
    STATE.unlink(missing_ok=True)
    lenz(
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

    for i in range(1, BUDGET + 1):
        candidates = lenz("suggest")
        config = candidates[0]["config"]
        metrics = evaluate(config)
        lenz("submit", "--config", json.dumps(config), "--metrics", json.dumps(metrics))
        best = lenz("incumbent")["metrics"]["y"]
        print(f"eval {i:2d}/{BUDGET}  y={metrics['y']:.4f}  best-so-far={best:.4f}  regret={best - BRANIN_MIN:.4f}")

    print(f"\nfinal incumbent: {lenz('incumbent')}")


if __name__ == "__main__":
    main()
