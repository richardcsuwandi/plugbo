"""Overlays best-so-far true-regret curves from multiple blind-test run
directories (each produced by `run_blind_test.py` or `run_blind_baseline.py`)
on one chart -- so different conditions (vanilla BO, sara+lenz,
sara+lenz+cake, ...) can be compared side by side. `sara-viz` only ever shows
one run at a time, so this fills the "compare conditions" gap without adding
a plotting dependency: plain inline SVG, matching the rest of this repo's
stdlib-only viz.

Usage:
    python3 -m benchmarks.plot_compare --root ./results/logs/hartmann6-compare

Expects `--root` to contain one subdirectory per condition (its name becomes
the legend label), each holding a `sandbox_<token>/state.json` +
`_answers/<token>.json` pair, i.e. exactly what `run_blind_test.py` /
`run_blind_baseline.py` write when pointed at `--root <that subdirectory>`.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from .functions import true_regret
from .obfuscate import ObfuscatedBenchmark

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D"]


def _pick_state(condition_dir: Path) -> Path | None:
    """Latest sandbox in this condition dir (by state.json mtime), not the
    lexicographically first leftover run.
    """
    candidates = [p for p in condition_dir.glob("sandbox_*/state.json") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_budget(run_dir: Path) -> int | None:
    """`run_meta.json`'s recorded budget for this sandbox, or None if there's
    no meta file or it never recorded one (e.g. very old runs)."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    budget = meta.get("budget")
    return budget if isinstance(budget, int) else None


def regret_trace_for_state(state_path: Path, condition_dir: Path | None = None) -> list[float] | None:
    """Best-so-far true regret for one sandbox state, capped at its budget.

    The cap matters for sandboxes recorded before the hard budget guardrail
    in `lenz/state.py` existed: those could over-run (e.g. an agent recording
    108 evaluations against a 100 budget), which would otherwise make that
    condition look unfairly strong against conditions that stayed in budget.
    Truncating here -- rather than editing the sandbox's state.json -- keeps
    the raw historical record intact while making every comparison plot fair.
    """
    condition_dir = condition_dir or state_path.parent.parent
    token = state_path.parent.name.removeprefix("sandbox_")
    secret_path = condition_dir / "_answers" / f"{token}.json"
    if not secret_path.exists():
        return None

    spec = ObfuscatedBenchmark.from_secret(json.loads(secret_path.read_text())).spec
    state = json.loads(state_path.read_text())
    constrained = spec.constraint_fn is not None

    best = float("inf")
    trace = []
    for t in state.get("trials", []):
        metrics = t.get("metrics") or {}
        if t.get("status") != "observed" or "y" not in metrics:
            continue
        # for constrained benchmarks, an infeasible trial's y is not a meaningful
        # regret -- skip it (mirrors run_blind_test.score_sandbox's feasibility gate)
        if constrained and not ("c" in metrics and float(metrics["c"]) <= spec.constraint_upper):
            continue
        best = min(best, true_regret(spec, float(metrics["y"])))
        trace.append(best)

    budget = _run_budget(state_path.parent)
    if budget is not None and len(trace) > budget:
        trace = trace[:budget]
    return trace


def regret_trace(condition_dir: Path) -> list[float] | None:
    """Best-so-far trace from the latest sandbox in a condition directory."""
    state_path = _pick_state(condition_dir)
    if state_path is None:
        return None
    return regret_trace_for_state(state_path, condition_dir)


def is_scorable(condition_dir: Path) -> bool:
    """True when this condition has a sandbox plus matching _answers secret."""
    state_path = _pick_state(condition_dir)
    if state_path is None:
        return False
    token = state_path.parent.name.removeprefix("sandbox_")
    return (condition_dir / "_answers" / f"{token}.json").is_file()


def _condition_status(n_evals: int, budget: int | None, meta_status: str | None) -> str:
    if meta_status == "running":
        return "running"
    if meta_status == "failed":
        return "failed"
    if budget is not None and n_evals < budget:
        return "stopped early"
    return "complete"


def condition_summary(condition_dir: Path) -> dict | None:
    """Per-condition stats for compare tables (trace plus eval-1 regret, status)."""
    trace = regret_trace(condition_dir)
    if not trace:
        return None

    state_path = _pick_state(condition_dir)
    run_dir = state_path.parent
    meta: dict | None = None
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = None

    budget = _run_budget(run_dir)
    meta_status = meta.get("status") if meta else None
    n_evals = len(trace)

    return {
        "name": condition_dir.name,
        "trace": trace,
        "regret_eval1": trace[0],
        "best_regret": trace[-1],
        "n_evals": n_evals,
        "budget": budget,
        "status": _condition_status(n_evals, budget, meta_status),
    }


def all_condition_summaries(condition_dir: Path) -> list[dict]:
    """One summary per scoreable sandbox, ordered by seed then start time."""
    summaries = []
    for state_path in condition_dir.glob("sandbox_*/state.json"):
        trace = regret_trace_for_state(state_path, condition_dir)
        if not trace:
            continue
        run_dir = state_path.parent
        meta = {}
        meta_path = run_dir / "run_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
        budget = _run_budget(run_dir)
        summaries.append(
            {
                "condition": condition_dir.name,
                "sandbox": run_dir.name,
                "seed": meta.get("seed"),
                "trace": trace,
                "best_regret": trace[-1],
                "n_evals": len(trace),
                "budget": budget,
                "status": _condition_status(len(trace), budget, meta.get("status")),
                "started_at": meta.get("started_at"),
                "meta": meta,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["seed"] is None,
            item["seed"] if item["seed"] is not None else 0,
            item["started_at"] or "",
        ),
    )


def _with_eval_zero(trace: list[float]) -> list[float]:
    """Prepend eval 0 (no observations yet) so every condition starts at the
    same point before the first evaluation diverges."""
    return [float("inf"), *trace]


def _legend_origin(
    corner: str,
    x0: float,
    y1: float,
    x1: float,
    y0: float,
    legend_w: float,
    legend_h: float,
    inset: float = 4.0,
) -> tuple[float, float]:
    left = x0 + inset if corner[1] == "l" else x1 - legend_w - inset
    top = y1 + inset if corner[0] == "t" else y0 - legend_h - inset
    return left, top


def pick_legend_corner(
    points: list[tuple[float, float]],
    x0: float,
    y1: float,
    x1: float,
    y0: float,
    legend_w: float,
    legend_h: float,
    inset: float = 4.0,
) -> str:
    """Choose the plot corner whose legend box covers the fewest curve points.

    Ties keep the earlier of tr, bl, br, tl so the default is the classic
    top-right overlay unless that corner sits on the curves (Ackley).
    """
    order = ("tr", "bl", "br", "tl")
    best = order[0]
    best_hits: int | None = None
    for corner in order:
        left, top = _legend_origin(corner, x0, y1, x1, y0, legend_w, legend_h, inset)
        right, bottom = left + legend_w, top + legend_h
        hits = sum(1 for x, y in points if left <= x <= right and top <= y <= bottom)
        if best_hits is None or hits < best_hits:
            best, best_hits = corner, hits
    return best



def collect_traces(root: Path) -> dict[str, list[float]]:
    traces = {}
    for condition_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trace = regret_trace(condition_dir)
        if trace:
            traces[condition_dir.name] = trace
    return traces


def _chart_svg(traces: dict[str, list[float]], width: int = 820, height: int = 440) -> str:
    pad_l, pad_r, pad_t, pad_b = 60, 20, 16, 36
    x0, y1 = pad_l, pad_t
    x1, y0 = width - pad_r, height - pad_b

    plot_traces = {label: _with_eval_zero(trace) for label, trace in traces.items()}
    finite_vals = [v for tr in plot_traces.values() for v in tr if math.isfinite(v)]
    y_min = min(0.0, min(finite_vals)) if finite_vals else 0.0
    y_max = max(finite_vals) if finite_vals else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    n_max = max(len(tr) for tr in plot_traces.values())

    def X(i: int) -> float:
        return x0 + (x1 - x0) * (i / max(n_max - 1, 1))

    def Y(v: float) -> float:
        if not math.isfinite(v):
            return y1
        return y1 + (y0 - y1) * (1 - (v - y_min) / (y_max - y_min))

    def pad_trace(trace: list[float]) -> list[float]:
        """Hold best-so-far flat after the last eval so early-stop runs stay visible."""
        if len(trace) >= n_max:
            return trace[:n_max]
        return trace + [trace[-1]] * (n_max - len(trace))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12">'
    ]
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="currentColor" stroke-opacity="0.35"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="currentColor" stroke-opacity="0.35"/>'
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = y_min + frac * (y_max - y_min)
        yy = Y(v)
        parts.append(
            f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="currentColor" stroke-opacity="0.08"/>'
        )
        parts.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="currentColor">{v:.2f}</text>')
    parts.append(f'<text x="{x0}" y="{y0 + 22}" fill="currentColor">0</text>')
    parts.append(f'<text x="{x1}" y="{y0 + 22}" text-anchor="end" fill="currentColor">{n_max - 1} evals</text>')

    legend_rows: list[tuple[str, str]] = []
    for label, trace in traces.items():
        n_obs = len(trace)
        eval_note = f", {n_obs} eval{'s' if n_obs != 1 else ''}" if n_obs < n_max - 1 else ""
        legend_rows.append((label, f"{label} (best {trace[-1]:.4f}{eval_note})"))

    for i, (label, trace) in enumerate(traces.items()):
        color = COLORS[i % len(COLORS)]
        n_obs = len(trace)
        padded = pad_trace(plot_traces[label])
        pts = " ".join(f"{X(j):.1f},{Y(v):.1f}" for j, v in enumerate(padded))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        mx, my = X(n_obs), Y(trace[-1])
        marker_r = 6 if n_obs == 1 else 4
        parts.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="{marker_r}" fill="{color}" '
            f'stroke="currentColor" stroke-width="1" stroke-opacity="0.4"/>'
        )

    # Legend sits in the emptiest corner so it does not cover still-high
    # curves (Ackley, top-right) or converged floors (Hartmann, bottom-right).
    legend_row_h = 18
    legend_pad = 6
    legend_w = 300
    legend_h = len(legend_rows) * legend_row_h + legend_pad * 2
    box_w = legend_w + legend_pad
    curve_pts = [
        (X(j), Y(v))
        for label, trace in traces.items()
        for j, v in enumerate(pad_trace(plot_traces[label]))
    ]
    corner = pick_legend_corner(curve_pts, x0, y1, x1, y0, box_w, legend_h)
    legend_left, legend_top = _legend_origin(corner, x0, y1, x1, y0, box_w, legend_h)
    parts.append(
        f'<rect x="{legend_left}" y="{legend_top}" width="{box_w}" height="{legend_h}" '
        f'rx="4" fill="currentColor" fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.20"/>'
    )
    legend_x = legend_left + legend_pad
    for i, (_, text) in enumerate(legend_rows):
        color = COLORS[i % len(COLORS)]
        cy = legend_top + legend_pad + i * legend_row_h + legend_row_h / 2
        parts.append(f'<circle cx="{legend_x}" cy="{cy:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 10}" y="{cy + 4:.1f}" fill="currentColor">{html.escape(text)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def build_html(traces: dict[str, list[float]], title: str) -> str:
    if not traces:
        body = "<p>No scoreable runs found under --root yet.</p>"
    else:
        body = _chart_svg(traces)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; padding: 24px; background: #0b0e14; color: #e6e6e6;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  @media (prefers-color-scheme: light) {{ body {{ background: #fff; color: #111; }} }}
  h1 {{ font-size: 16px; font-weight: 600; margin: 0 0 16px; }}
  svg {{ max-width: 100%; height: auto; }}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
{body}
</body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay best-so-far regret curves from multiple blind-test run dirs.")
    p.add_argument("--root", required=True, help="parent dir; each subdirectory is one labeled condition")
    p.add_argument("--out", default=None, help="output HTML path (default: <root>/compare.html)")
    p.add_argument("--title", default="Blind-test comparison")
    args = p.parse_args()

    root = Path(args.root)
    traces = collect_traces(root)
    out = Path(args.out) if args.out else root / "compare.html"
    out.write_text(build_html(traces, args.title))

    if not traces:
        print(f"No scoreable runs found under {root} yet -- wrote an empty page to {out}.")
        return
    print(f"Wrote {out} ({len(traces)} condition(s)):")
    for label, trace in traces.items():
        print(f"  {label}: {len(trace)} evals, best regret {trace[-1]:.6f}")


if __name__ == "__main__":
    main()
