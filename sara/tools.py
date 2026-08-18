"""Sara's only two tools: `bash` (to call the lenz CLI and run evaluations)
and `read` (to inspect files), both sandboxed to the run's working directory.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

BASH_TIMEOUT = 120

TOOL_SCHEMAS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the sandbox working directory. Use this to call "
            "the lenz CLI and to run the evaluation command."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"cmd": {"type": "string", "description": "the shell command to run"}},
            "required": ["cmd"],
        },
    },
    {
        "name": "read",
        "description": "Read a file's contents, relative to the sandbox working directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "path relative to the sandbox"}},
            "required": ["path"],
        },
    },
]


def make_bash_tool(sandbox: Path) -> Callable[..., str]:
    def run(cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=sandbox, capture_output=True, text=True, timeout=BASH_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            return f"[error] command timed out after {BASH_TIMEOUT}s"
        out = result.stdout
        if result.stderr:
            out += ("\n" if out else "") + f"[stderr]\n{result.stderr}"
        if result.returncode != 0:
            out += f"\n[exit code {result.returncode}]"
        return out or "[no output]"

    return run


def make_read_tool(sandbox: Path) -> Callable[..., str]:
    def run(path: str) -> str:
        target = (sandbox / path).resolve()
        try:
            target.relative_to(sandbox.resolve())
        except ValueError:
            return f"[error] path '{path}' is outside the sandbox"
        if not target.exists():
            return f"[error] file not found: {path}"
        try:
            return target.read_text()
        except UnicodeDecodeError:
            return f"[error] cannot read binary file: {path}"

    return run


def build_tools(sandbox: Path) -> tuple[list[dict], dict[str, Callable[..., str]]]:
    sandbox.mkdir(parents=True, exist_ok=True)
    handlers = {"bash": make_bash_tool(sandbox), "read": make_read_tool(sandbox)}
    return TOOL_SCHEMAS, handlers
