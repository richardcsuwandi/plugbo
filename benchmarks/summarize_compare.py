"""Aggregate repeated benchmark conditions into a machine-readable summary."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from .plot_compare import all_condition_summaries

CHECKPOINTS = (25, 50, 100)


def _bootstrap_median_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    draws = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(n_boot)
    )
    return [draws[int(0.025 * n_boot)], draws[min(n_boot - 1, int(0.975 * n_boot))]]


def _log_regret_auc(trace: list[float], floor: float = 1e-12) -> float:
    logs = [math.log10(max(float(value), floor)) for value in trace]
    if len(logs) == 1:
        return logs[0]
    return sum((left + right) / 2 for left, right in zip(logs, logs[1:])) / (len(logs) - 1)


def summarize_condition(condition_dir: Path) -> dict | None:
    runs = all_condition_summaries(condition_dir)
    if not runs:
        return None
    finals = [run["best_regret"] for run in runs]
    aucs = [_log_regret_auc(run["trace"]) for run in runs]
    checkpoints = {}
    for checkpoint in CHECKPOINTS:
        values = [run["trace"][checkpoint - 1] for run in runs if len(run["trace"]) >= checkpoint]
        checkpoints[str(checkpoint)] = {
            "n": len(values),
            "median": statistics.median(values) if values else None,
            "median_ci95": _bootstrap_median_ci(values),
        }
    return {
        "condition": condition_dir.name,
        "n_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "n_complete": sum(run["status"] == "complete" for run in runs),
        "n_running": sum(run["status"] == "running" for run in runs),
        "n_stopped_early": sum(run["status"] == "stopped early" for run in runs),
        "final_regret": {
            "median": statistics.median(finals),
            "median_ci95": _bootstrap_median_ci(finals),
        },
        "log_regret_auc": {
            "median": statistics.median(aucs),
            "median_ci95": _bootstrap_median_ci(aucs),
        },
        "checkpoints": checkpoints,
        "runs": [
            {
                key: run[key]
                for key in ("sandbox", "seed", "best_regret", "n_evals", "budget", "status")
            }
            for run in runs
        ],
    }


def summarize_root(root: Path) -> dict:
    conditions = []
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = summarize_condition(condition_dir)
        if summary is not None:
            conditions.append(summary)
    return {"root": str(root), "conditions": conditions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="comparison root containing condition directories")
    parser.add_argument("--out", default=None, help="default: <root>/summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "summary.json"
    payload = summarize_root(root)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out} ({len(payload['conditions'])} conditions)")


if __name__ == "__main__":
    main()
