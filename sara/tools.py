"""Sara's only two tools: `bash` (to call the lenz CLI and run evaluations)
and `read` (to inspect files), both sandboxed to the run's working directory.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

BASH_TIMEOUT = 120

# Matches the lenz CLI, `python -m lenz`, and `import lenz` / `from lenz`.
_LENZ_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])lenz(?![A-Za-z0-9_])")

TOOL_SCHEMAS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the sandbox working directory. Use this to call "
            "the evaluation command and inspect the working directory."
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


def _path_without_lenz_bin(env: dict[str, str]) -> dict[str, str]:
    """Drop PATH entries that contain a `lenz` executable so `lenz` is not callable."""
    out = dict(env)
    kept = []
    for part in out.get("PATH", "").split(os.pathsep):
        if part and (Path(part) / "lenz").exists():
            continue
        kept.append(part)
    out["PATH"] = os.pathsep.join(kept)
    return out


def _ensure_import_shadow(sandbox: Path) -> Path:
    """A PYTHONPATH prefix whose `lenz` package always raises, so `import lenz`
    fails even when the command string avoids the name (e.g. `'le'+'nz'`).
    """
    shadow = sandbox / ".no_optimizer"
    pkg = shadow / "lenz"
    pkg.mkdir(parents=True, exist_ok=True)
    init = pkg / "__init__.py"
    if not init.exists():
        init.write_text("raise ImportError('this run has no external optimizer')\n")
    return shadow


def make_bash_tool(sandbox: Path, *, block_lenz: bool = False) -> Callable[..., str]:
    def run(cmd: str) -> str:
        if block_lenz and _LENZ_RE.search(cmd or ""):
            return (
                "[error] this run has no external optimizer. Propose configs yourself "
                "and evaluate them with ./oracle. That command is not available."
            )
        env = os.environ.copy()
        if block_lenz:
            env = _path_without_lenz_bin(env)
            shadow = _ensure_import_shadow(sandbox)
            env["PYTHONPATH"] = str(shadow) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=BASH_TIMEOUT,
                env=env,
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


def build_tools(sandbox: Path, *, block_lenz: bool = False) -> tuple[list[dict], dict[str, Callable[..., str]]]:
    sandbox.mkdir(parents=True, exist_ok=True)
    schemas = TOOL_SCHEMAS
    if block_lenz:
        schemas = [
            {
                **TOOL_SCHEMAS[0],
                "description": (
                    "Run a shell command in the sandbox working directory. Use this to "
                    "run ./oracle and inspect files. There is no external optimizer CLI."
                ),
            },
            TOOL_SCHEMAS[1],
        ]
    handlers = {
        "bash": make_bash_tool(sandbox, block_lenz=block_lenz),
        "read": make_read_tool(sandbox),
    }
    return schemas, handlers
