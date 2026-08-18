"""A minimal local viewer for agentic-bo run artifacts (state.json,
trace.jsonl, run_meta.json) under `--root` (default: results/logs). Stdlib
only, no new dependencies: `python3 -m viz.server`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from benchmarks.plot_all import find_groups
from benchmarks.plot_compare import _pick_state, collect_traces, condition_summary, is_scorable

from .captions import classify_group, classify_relpath, experiment_caption

STATIC_DIR = Path(__file__).parent / "static"


def _load_meta(run_dir: Path) -> dict | None:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _parse_iso(ts: str | float | int | None) -> float | None:
    if ts is None or ts == "":
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _started_at_epoch(run_dir: Path, state: dict, meta: dict | None) -> float | None:
    if meta and meta.get("started_at"):
        parsed = _parse_iso(meta["started_at"])
        if parsed is not None:
            return parsed
    trials = state.get("trials") or []
    times = [
        t.get("created_at")
        for t in trials
        if isinstance(t, dict) and t.get("created_at")
    ]
    if times:
        return min(times)
    state_path = run_dir / "state.json"
    return state_path.stat().st_mtime if state_path.exists() else None


def _duration_seconds(state: dict, meta: dict | None, trace: list[dict]) -> float | None:
    if meta and meta.get("started_at"):
        start = _parse_iso(meta["started_at"])
        end = _parse_iso(meta.get("ended_at")) if meta.get("ended_at") else None
        if start is not None and end is not None:
            return max(0.0, end - start)
    if trace:
        tss = [e["ts"] for e in trace if isinstance(e.get("ts"), (int, float))]
        if len(tss) >= 2:
            return max(0.0, max(tss) - min(tss))
    times = [t.get("observed_at") for t in state.get("trials", []) if t.get("observed_at")]
    starts = [t.get("created_at") for t in state.get("trials", []) if t.get("created_at")]
    if times and starts:
        return max(0.0, max(times) - min(starts))
    return None


def _epoch_to_iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _run_kind(meta: dict | None, has_trace: bool) -> str:
    """'sara' (agent-driven) vs 'lenz' (scripted/manual, no agent).

    Blind sara runs historically wrote a trace but no run_meta.json; vanilla
    BO writes neither. An explicit meta['kind'] wins so a lenz baseline can
    still record timestamps without being mislabeled as an agent run.
    """
    kind = (meta or {}).get("kind")
    if kind in ("sara", "sara-only", "sara-noblind"):
        return "sara"
    if kind == "lenz":
        return "lenz"
    return "sara" if (meta is not None or has_trace) else "lenz"


def find_runs(root: Path) -> list[dict]:
    root = root.resolve()
    if not root.exists():
        return []
    state_paths = list(root.rglob("state.json"))
    runs = []
    seen = set()
    for state_path in sorted(state_paths):
        run_dir = state_path.parent
        rel = run_dir.relative_to(root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            state = json.loads(state_path.read_text())
            meta = _load_meta(run_dir)
            has_trace = (run_dir / "trace.jsonl").exists()
            trials = state.get("trials") or []
            n_observed = sum(1 for t in trials if isinstance(t, dict) and t.get("status") == "observed")
            tax = classify_relpath(rel)
            shelf = state.get("shelf") if isinstance(state.get("shelf"), dict) else {}
            objectives = shelf.get("objectives") or []
            runs.append(
                {
                    "name": rel,
                    "n_trials": len(trials),
                    "n_observed": n_observed,
                    "surrogate": shelf.get("surrogate", "fixed"),
                    "is_moo": len(objectives) > 1,
                    "has_trace": has_trace,
                    "run_kind": _run_kind(meta, has_trace),
                    "provider": meta.get("provider") if meta else None,
                    "model": meta.get("model") if meta else None,
                    "status": meta.get("status") if meta else ("completed" if trials else "empty"),
                    "started_at": (meta.get("started_at") if meta else None)
                    or _epoch_to_iso(_started_at_epoch(run_dir, state, meta)),
                    "group": tax["group"],
                    "condition": tax["condition"],
                    "benchmark": tax["benchmark"],
                    "benchmark_label": tax["benchmark_label"],
                    "backend": tax["backend"],
                    "backend_label": tax["backend_label"],
                    "disclosure": tax["disclosure"],
                    "disclosure_label": tax["disclosure_label"],
                    "heading": tax["heading"],
                    "_sort_key": _started_at_epoch(run_dir, state, meta) or 0,
                }
            )
        except Exception:
            continue
    runs.sort(key=lambda r: r["_sort_key"], reverse=True)
    for r in runs:
        del r["_sort_key"]
    return runs


def _load_trace(run_dir: Path) -> list[dict]:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        return []
    entries = []
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"role": "?", "content": line})
    return entries


def _compute_convergence(state: dict) -> dict | None:
    objectives = state.get("shelf", {}).get("objectives", [])
    if len(objectives) != 1:
        return None
    obj = objectives[0]
    metric, minimize = obj["metric"], obj["minimize"]
    observed = [t for t in state.get("trials", []) if t.get("status") == "observed" and t.get("metrics")]
    observed.sort(key=lambda t: t.get("observed_at") or 0)

    points = []
    best = None
    for i, t in enumerate(observed, start=1):
        if metric not in t["metrics"]:
            continue
        value = float(t["metrics"][metric])
        if best is None or (value < best if minimize else value > best):
            best = value
        points.append({"i": i, "trial_id": t["trial_id"], "value": value, "best": best})
    return {"metric": metric, "minimize": minimize, "points": points}


def _kernel_generations(state: dict) -> list[dict]:
    events = state.get("events", [])
    return [e for e in events if e.get("command") == "evolve-kernels" and "generation" in e and "population" in e]


def _cake_blob(state: dict) -> dict:
    plugins = state.get("plugins") or {}
    if "cake" in plugins:
        return plugins["cake"]
    shelf = state.get("shelf") or {}
    return {
        "kernel_populations": shelf.get("kernel_populations") or {},
        "kernel_evolution_states": shelf.get("kernel_evolution_states") or {},
        "kernel_population": shelf.get("kernel_population") or [],
        "kernel_evolution_state": shelf.get("kernel_evolution_state"),
        "kernel_population_size": shelf.get("kernel_population_size"),
    }


_RECONFIG_COMMANDS = {
    "set-acqf",
    "set-bounds",
    "set-objectives",
    "set-constraints",
    "set-surrogate",
    "set-region",
    "set-sampler",
    "set-belief",
}


def _reconfigurations(state: dict) -> list[dict]:
    """Every mid-campaign backend reconfiguration Sara made (mid-run problem
    reformulation), in order, with the new value each
    call set -- `lenz`'s state.json events already carry that payload
    (`log_event("set-acqf", acqf=...)` etc. in lenz/commands.py), so no
    trace parsing is needed here.
    """
    return [e for e in state.get("events", []) if e.get("command") in _RECONFIG_COMMANDS]


# Every `lenz <subcommand>` invocation inside one bash `cmd` string. Read-only
# introspection commands (status/diagnostics/predict/score/trials/incumbent)
# never touch state.json's own event log (lenz/commands.py only calls
# `log_event` for mutating commands), so trace.jsonl -- where the agent's
# actual bash invocations live -- is the only place to observe them at all.
_LENZ_CALL_RE = re.compile(r"\blenz\s+([a-z][a-z-]*)\b")


def _classify_lenz_calls(cmd: str) -> list[str]:
    """`suggest` is split into `suggest`/`suggest_bounded`/`suggest_around`
    by its flags. Handles
    multiple `lenz` invocations chained in one bash command (e.g. `lenz
    suggest ... && lenz submit ...`) by scoping each call's flag search to
    the text between it and the next `lenz` invocation.
    """
    matches = list(_LENZ_CALL_RE.finditer(cmd))
    calls = []
    for i, m in enumerate(matches):
        subcmd = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cmd)
        segment = cmd[m.end() : end]
        if subcmd == "suggest":
            if re.search(r"--around\b", segment):
                subcmd = "suggest_around"
            elif re.search(r"--bounds\b", segment):
                subcmd = "suggest_bounded"
        calls.append(subcmd)
    return calls


def _tool_use_events(state: dict, trace: list[dict]) -> list[dict] | None:
    """One entry per `lenz` subcommand invocation across the whole trace,
    each tagged with its call type and the "normalized trial progress"
    (fraction of this run's final observed-trial count already landed) at
    the moment the agent issued it -- the raw material for the "relative
    frequency of lenz calls ... over normalized trial progress". `None` if
    there's no trace, or no observed trials to normalize progress against.

    Sobol warm-start evaluations happen via `benchmarks.lenz_loop` before
    `sara run_campaign` starts and so never appear in trace.jsonl -- which
    is correct here: this is specifically about the agent's own tool use,
    not the shared warm-start every condition gets.
    """
    observed_ts = sorted(
        t["observed_at"]
        for t in state.get("trials", [])
        if t.get("status") == "observed" and t.get("observed_at") is not None
    )
    total = len(observed_ts)
    if not trace or total == 0:
        return None

    events = []
    for entry in trace:
        if entry.get("role") != "assistant":
            continue
        ts = entry.get("ts")
        progress = None
        if isinstance(ts, (int, float)):
            n_before = sum(1 for o in observed_ts if o <= ts)
            progress = min(1.0, n_before / total)
        for tc in entry.get("tool_calls") or []:
            if tc.get("name") != "bash":
                continue
            cmd = (tc.get("arguments") or {}).get("cmd") or ""
            if "lenz" not in cmd:
                continue
            for call_type in _classify_lenz_calls(cmd):
                events.append({"call_type": call_type, "progress": progress})
    return events or None


def run_detail(root: Path, name: str) -> dict | None:
    root = root.resolve()
    run_dir = (root / name).resolve()
    if not (run_dir.is_relative_to(root) and (run_dir / "state.json").exists()):
        return None
    state = json.loads((run_dir / "state.json").read_text())
    meta = _load_meta(run_dir)
    trace = _load_trace(run_dir)
    has_trace = (run_dir / "trace.jsonl").exists()
    return {
        "name": name,
        "state": state,
        "meta": meta,
        "run_kind": _run_kind(meta, has_trace),
        "trace": trace,
        "convergence": _compute_convergence(state),
        "kernel_generations": _kernel_generations(state),
        "tool_use": _tool_use_events(state, trace),
        "reconfigurations": _reconfigurations(state),
        "started_at": (meta.get("started_at") if meta else None)
        or _epoch_to_iso(_started_at_epoch(run_dir, state, meta)),
        "duration_seconds": _duration_seconds(state, meta, trace),
    }


def _sandbox_started_at(sandbox_dir: Path) -> float | None:
    """Same timestamp fallback as a single run: meta, then trials, then mtime."""
    meta = _load_meta(sandbox_dir)
    state: dict = {}
    state_path = sandbox_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
    return _started_at_epoch(sandbox_dir, state, meta)


def _group_started_at(child_dirs: list[Path]) -> float | None:
    keys = []
    for child in child_dirs:
        for sandbox in child.glob("sandbox_*"):
            if sandbox.is_dir():
                started = _sandbox_started_at(sandbox)
                if started is not None:
                    keys.append(started)
    return max(keys) if keys else None


def find_compare_groups(root: Path) -> list[dict]:
    """Comparison groups: dirs whose children each hold sandbox_*/state.json."""
    root = root.resolve()
    groups = []
    for g in find_groups(root):
        rel = g.relative_to(root).as_posix()
        child_dirs = [c for c in g.iterdir() if c.is_dir() and any(c.glob("sandbox_*"))]
        tax = classify_group(rel)
        backends: set[str] = set()
        disclosures: set[str] = set()
        statuses: set[str] = set()
        if tax["backend"]:
            backends.add(tax["backend"])
        if tax["disclosure"]:
            disclosures.add(tax["disclosure"])
        for child in child_dirs:
            child_tax = classify_relpath(f"{rel}/{child.name}")
            if child_tax["backend"]:
                backends.add(child_tax["backend"])
            if child_tax["disclosure"]:
                disclosures.add(child_tax["disclosure"])
            metas = list(child.glob("sandbox_*/run_meta.json"))
            if metas:
                latest = max(metas, key=lambda p: p.stat().st_mtime)
                try:
                    meta = json.loads(latest.read_text())
                except (json.JSONDecodeError, OSError):
                    meta = {}
                statuses.add(meta.get("status") or "completed")
            else:
                statuses.add("completed")
        groups.append(
            {
                "name": rel,
                "title": rel.replace("/", " / "),
                "heading": tax["heading"],
                "caption": experiment_caption(rel),
                "n_conditions": len(child_dirs),
                "n_scored": sum(1 for c in child_dirs if is_scorable(c)),
                "benchmark": tax["benchmark"],
                "benchmark_label": tax["benchmark_label"],
                "axis": tax["axis"],
                "backends": sorted(backends),
                "disclosures": sorted(disclosures),
                "statuses": sorted(statuses),
                "started_at": _epoch_to_iso(_group_started_at(child_dirs)),
            }
        )
    groups.sort(key=lambda x: x["name"])
    return groups


def compare_group_detail(root: Path, name: str) -> dict | None:
    root = root.resolve()
    group_dir = (root / name).resolve()
    if not (group_dir.is_relative_to(root) and group_dir.is_dir()):
        return None
    traces = collect_traces(group_dir)
    conditions = []
    for condition_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        summary = condition_summary(condition_dir)
        if summary:
            item = {k: v for k, v in summary.items() if k != "trace"}
            state_path = _pick_state(condition_dir)
            if state_path is not None:
                item["run_name"] = state_path.parent.relative_to(root).as_posix()
            conditions.append(item)
    return {
        "name": name,
        "title": name.replace("/", " / "),
        "caption": experiment_caption(name),
        "traces": traces,
        "conditions": conditions,
    }


def delete_run(root: Path, name: str) -> bool:
    """Permanently removes a run's directory (state.json, trace.jsonl, ...).
    Returns False (no-op) for a name that doesn't resolve to a real run
    inside `root` -- callers should treat that as 404, not 500.
    """
    root = root.resolve()
    run_dir = (root / name).resolve()
    if not (run_dir.is_relative_to(root) and run_dir != root and (run_dir / "state.json").exists()):
        return False
    shutil.rmtree(run_dir)
    return True


class Handler(BaseHTTPRequestHandler):
    root: Path = Path("results/logs")

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        target = (STATIC_DIR / rel_path).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.exists():
            self.send_response(404)
            self.end_headers()
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/runs":
            try:
                self._send_json(find_runs(self.root))
            except Exception as exc:
                self._send_json({"error": f"failed to list runs: {exc}"}, status=500)
        elif path == "/api/compare-groups":
            self._send_json(find_compare_groups(self.root))
        elif path.startswith("/api/compare-groups/"):
            name = unquote(path[len("/api/compare-groups/") :])
            detail = compare_group_detail(self.root, name)
            if detail is None:
                self._send_json({"error": "compare group not found"}, status=404)
            else:
                self._send_json(detail)
        elif path.startswith("/groups/") and path.endswith("/compare.html"):
            rel = unquote(path[len("/groups/") : -len("/compare.html")])
            target = (self.root / rel / "compare.html").resolve()
            root = self.root.resolve()
            if not (target.is_relative_to(root) and target.is_file()):
                self.send_response(404)
                self.end_headers()
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/api/runs/"):
            name = unquote(path[len("/api/runs/") :])
            detail = run_detail(self.root, name)
            if detail is None:
                self._send_json({"error": "run not found"}, status=404)
            else:
                self._send_json(detail)
        elif path == "/" or path == "":
            self._send_static("index.html")
        else:
            self._send_static(path.lstrip("/"))

    def do_DELETE(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if not path.startswith("/api/runs/"):
            self.send_response(404)
            self.end_headers()
            return
        name = unquote(path[len("/api/runs/") :])
        if delete_run(self.root, name):
            self._send_json({"ok": True})
        else:
            self._send_json({"ok": False, "error": "run not found"}, status=404)


def main() -> None:
    p = argparse.ArgumentParser(description="Local viewer for agentic-bo run artifacts.")
    p.add_argument("--root", default="results/logs", help="directory to scan for runs (default: results/logs)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = p.parse_args()

    Handler.root = Path(args.root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"agentic-bo viewer serving {Handler.root.resolve()} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
