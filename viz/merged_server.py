"""A second, standalone viewer: overlays mean +/- standard-error regret
bands across every seed found for each condition in a compare group,
instead of `sara-viz`'s one-seed-at-a-time view. Stdlib only, no new
dependencies: `python3 -m viz.merged_server`.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from benchmarks.plot_all import find_groups
from benchmarks.plot_compare import group_seeds

from .aggregate import aggregate_group
from .captions import classify_group, experiment_caption

STATIC_DIR = Path(__file__).parent / "static_merged"


def find_merge_groups(root: Path) -> list[dict]:
    """One entry per compare group (all seeds pooled, unlike sara-viz's
    Experiments tab which slices seed 42/43/44 into separate rows)."""
    root = root.resolve()
    groups = []
    for g in find_groups(root):
        rel = g.relative_to(root).as_posix()
        seeds = group_seeds(g)
        if len(seeds) < 2:
            continue
        tax = classify_group(rel)
        groups.append(
            {
                "name": rel,
                "heading": tax["heading"],
                "caption": experiment_caption(rel),
                "benchmark": tax["benchmark"],
                "benchmark_label": tax["benchmark_label"],
                "seeds": seeds,
            }
        )
    groups.sort(key=lambda x: x["name"])
    return groups


def merge_group_detail(root: Path, name: str) -> dict | None:
    root = root.resolve()
    group_dir = (root / name).resolve()
    if not (group_dir.is_relative_to(root) and group_dir.is_dir()):
        return None
    conditions = aggregate_group(group_dir)
    tax = classify_group(name)
    return {
        "name": name,
        "heading": tax["heading"],
        "caption": experiment_caption(name),
        "conditions": conditions,
    }


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
        if path == "/api/merge-groups":
            self._send_json(find_merge_groups(self.root))
        elif path.startswith("/api/merge-groups/"):
            name = unquote(path[len("/api/merge-groups/") :])
            detail = merge_group_detail(self.root, name)
            if detail is None:
                self._send_json({"error": "compare group not found"}, status=404)
            else:
                self._send_json(detail)
        elif path == "/" or path == "":
            self._send_static("index.html")
        else:
            self._send_static(path.lstrip("/"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cross-seed mean +/- standard-error viewer for plugbo compare groups."
    )
    p.add_argument("--root", default="results/logs", help="directory to scan for compare groups (default: results/logs)")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = p.parse_args()

    Handler.root = Path(args.root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"plugbo merged viewer serving {Handler.root.resolve()} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
