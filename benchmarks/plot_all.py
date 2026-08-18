"""Regenerate every comparison group's `compare.html` across the run tree in
one pass -- the multi-experiment counterpart to calling `plot_compare.py` by
hand on one `--root` at a time.

Auto-discovers "comparison group" directories: any directory whose immediate
subdirectories each hold a `sandbox_<token>/state.json` -- exactly the shape
`run_synthetic.sh` / `run_bolt.sh` already produce (one subdirectory per condition).
Each group gets its OWN `compare.html` written in its OWN directory, via
`plot_compare`'s own `collect_traces`/`build_html` -- this script does not
merge different groups into one chart (see docs/observations.md: "Do not
average blind and no-blind curves. They are different experiments.").

Usage:
    python3 -m benchmarks.plot_all
    python3 -m benchmarks.plot_all --root ./results/logs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plot_compare import build_html, collect_traces


def _is_condition_dir(p: Path) -> bool:
    return any(s.is_dir() for s in p.glob("sandbox_*"))


def find_groups(root: Path) -> list[Path]:
    """Every directory under `root` that looks like a plot_compare `--root`:
    at least one immediate child is itself a condition directory. Sorted so
    output is stable and shallow groups print before nested ones.
    """
    root = root.resolve()
    groups = []
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        children = [c for c in d.iterdir() if c.is_dir()]
        if children and any(_is_condition_dir(c) for c in children):
            groups.append(d)
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate compare.html for every comparison group under --root.")
    p.add_argument("--root", default="results/logs", help="directory tree to scan (default: results/logs)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    groups = find_groups(root)
    if not groups:
        print(f"No comparison groups found under {root}")
        return

    print(f"Found {len(groups)} comparison group(s) under {root}:\n")
    for g in groups:
        traces = collect_traces(g)
        title = str(g.relative_to(root)).replace("/", " / ")
        out = g / "compare.html"
        out.write_text(build_html(traces, title))
        rel = g.relative_to(root)
        if traces:
            print(f"  {rel}  ({len(traces)} condition(s)) -> {out}")
            for label, tr in traces.items():
                print(f"      {label}: {len(tr)} evals, best regret {tr[-1]:.6f}")
        else:
            print(f"  {rel}  (no scoreable runs yet) -> {out}")
        print()

    print(f"Done -- {len(groups)} compare.html file(s) written/refreshed, each in its own group directory.")


if __name__ == "__main__":
    main()
