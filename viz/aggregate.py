"""Cross-seed aggregation for compare groups: turns per-seed best-so-far
regret traces (results/logs/<group>/<condition>/sandbox_<seed-token>/) into
one mean +/- standard-error curve per condition, so a group with seeds
42/43/44 renders as a single band instead of three overlaid single-seed
lines. Reuses the scoring in benchmarks.plot_compare rather than
re-deriving it, so a mean curve here always matches what a single-seed
`sara-viz` compare view would show for the same sandbox.
"""

from __future__ import annotations

import math
from pathlib import Path

from benchmarks.plot_compare import (
    _condition_status,
    _run_budget,
    _run_meta_status,
    _sandbox_seed,
    regret_trace_for_state,
)


def _pad_flat(trace: list[float], n: int) -> list[float]:
    """Hold best-so-far flat past an early stop, matching plot_compare's
    single-seed chart so an early-stopping seed doesn't drag the mean down
    at evals it never reached."""
    if len(trace) >= n:
        return trace[:n]
    return trace + [trace[-1]] * (n - len(trace))


def _latest_run_dir_by_seed(condition_dir: Path) -> dict[int, Path]:
    """One sandbox_<token> run dir per seed, preferring a completed run over
    an in-flight rerun of the same seed (by mtime among same-status
    candidates). Without this, kicking off a seed-45/46 backfill while a
    prior seed is still mid-campaign would silently blend that partial
    trace into the mean the moment its sandbox_* directory appears --
    e.g. rerunning seed 44 while 42/43 are already complete must not let
    the fresh, still-running seed-44 sandbox shadow a finished one."""
    candidates: dict[int, list[tuple[Path, float, bool]]] = {}
    for state_path in condition_dir.glob("sandbox_*/state.json"):
        run_dir = state_path.parent
        seed = _sandbox_seed(run_dir)
        if seed is None:
            continue
        mtime = state_path.stat().st_mtime
        completed = _run_meta_status(run_dir) == "completed"
        candidates.setdefault(seed, []).append((run_dir, mtime, completed))

    latest: dict[int, Path] = {}
    for seed, entries in candidates.items():
        completed_entries = [e for e in entries if e[2]]
        pool = completed_entries or entries
        run_dir, _mtime, _completed = max(pool, key=lambda e: e[1])
        latest[seed] = run_dir
    return latest


def condition_seed_traces(condition_dir: Path) -> dict[int, list[float]]:
    """One regret trace per seed for this condition (latest sandbox when a
    seed has more than one, e.g. a rerun)."""
    traces = {}
    for seed, run_dir in _latest_run_dir_by_seed(condition_dir).items():
        trace = regret_trace_for_state(run_dir / "state.json", condition_dir)
        if trace:
            traces[seed] = trace
    return traces


def aggregate_condition(condition_dir: Path) -> dict | None:
    """Mean +/- standard error of best-so-far regret across every *finished*
    seed found for one condition. `None` when no seed has a scoreable
    sandbox. A seed still `running` (or `failed`) is excluded from the mean
    entirely rather than folded in as a partial trace -- a live backfill job
    a few evals into a fresh seed would otherwise silently pull the mean
    toward whatever its early, unrepresentative trace happens to show."""
    seed_traces = condition_seed_traces(condition_dir)
    if not seed_traces:
        return None

    all_seeds = sorted(seed_traces)
    run_dirs = _latest_run_dir_by_seed(condition_dir)

    per_seed_n_evals = {s: len(seed_traces[s]) for s in all_seeds}
    per_seed_status: dict[int, str] = {}
    budget = None
    for s in all_seeds:
        run_dir = run_dirs.get(s)
        b = _run_budget(run_dir) if run_dir is not None else None
        if b is not None:
            budget = b
        meta_status = _run_meta_status(run_dir) if run_dir is not None else None
        per_seed_status[s] = _condition_status(per_seed_n_evals[s], b, meta_status)

    # "stopped early" is a valid, finished result (the agent reported its
    # incumbent and stopped, e.g. after the 1.5x-budget soft nudge) -- it
    # counts the same as a full-budget run here. Only "failed" and "running"
    # mean the seed doesn't have a final answer yet.
    seeds = [s for s in all_seeds if per_seed_status[s] not in ("failed", "running")]
    n_failed = sum(1 for s in all_seeds if per_seed_status[s] == "failed")
    n_running = sum(1 for s in all_seeds if per_seed_status[s] == "running")
    n_seeds = len(seeds)

    if n_seeds == 0:
        return None

    n_max = max(len(seed_traces[s]) for s in seeds)
    padded = [_pad_flat(seed_traces[s], n_max) for s in seeds]

    mean: list[float] = []
    stderr: list[float] = []
    for i in range(n_max):
        vals = [tr[i] for tr in padded]
        m = sum(vals) / n_seeds
        if n_seeds > 1:
            var = sum((v - m) ** 2 for v in vals) / (n_seeds - 1)
            se = math.sqrt(var / n_seeds)
        else:
            se = 0.0
        mean.append(m)
        stderr.append(se)

    per_seed_best = {s: seed_traces[s][-1] for s in seeds}

    if n_failed == 0 and n_running == 0:
        status = "complete"
    elif n_failed and n_running:
        status = f"{n_seeds} seeds complete ({n_failed} failed, {n_running} running)"
    elif n_failed:
        status = f"{n_seeds} seeds complete ({n_failed} failed)"
    else:
        status = f"{n_seeds} seeds complete ({n_running} running)"

    return {
        "name": condition_dir.name,
        "seeds": seeds,
        "n_seeds": n_seeds,
        "mean": mean,
        "stderr": stderr,
        "best_mean": mean[-1],
        "best_stderr": stderr[-1],
        "per_seed_best": per_seed_best,
        "per_seed_n_evals": per_seed_n_evals,
        "budget": budget,
        "status": status,
    }


def aggregate_group(group_dir: Path) -> list[dict]:
    """One aggregate_condition() result per condition subdirectory, skipping
    condition dirs with nothing scoreable (e.g. an empty `_answers`)."""
    conditions = []
    for condition_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        if condition_dir.name == "_answers":
            continue
        agg = aggregate_condition(condition_dir)
        if agg is not None:
            conditions.append(agg)
    return conditions
