"""Sara-only (no lenz) harness helpers: trial log + oracle wrapper.

The agent proposes configs and runs `./oracle`. The wrapper records each
hit into `state.json` in the same shape `score_sandbox` already reads, so
plots stay comparable. Lenz is never created and never on the agent's PATH.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

# Baked in at generation time, not read from `sys.executable` inside the
# wrapper: the agent's `bash` tool runs `./oracle` under whatever `python3`
# is first on its PATH, which is not guaranteed to be the venv that has
# `benchmarks` (and, for surrogate oracles, `torch`/`huggingface_hub`)
# installed. A self-contained formula oracle happens to need nothing but the
# stdlib, so this was invisible until `bolt_lora`'s surrogate oracle (which
# does `from benchmarks.bolt_lora import ...`) hit a bare `env python3` and
# crashed with `ModuleNotFoundError: No module named 'benchmarks'`.
_WRAPPER = r'''#!{interpreter}
"""Black-box eval wrapper: call the real oracle and append to ./state.json."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

SANDBOX = Path(__file__).resolve().parent
IMPL = SANDBOX / ".oracle_impl"
STATE = SANDBOX / "state.json"
BUDGET = {budget}
# Same interpreter this wrapper was generated under -- guaranteed to have
# `.oracle_impl`'s dependencies installed, regardless of what `python3`
# resolves to in the agent's shell environment.
INTERPRETER = {interpreter!r}


def _n_observed(state: dict) -> int:
    return sum(1 for t in state.get("trials", []) if t.get("status") == "observed")


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({{"error": "usage: ./oracle '<config-json>'"}}))
        sys.exit(1)
    try:
        config = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({{"error": f"malformed config JSON: {{e}}"}}))
        sys.exit(1)
    if not IMPL.is_file():
        print(json.dumps({{"error": "missing .oracle_impl"}}))
        sys.exit(1)

    state = json.loads(STATE.read_text()) if STATE.is_file() else {{"trials": []}}
    if BUDGET and _n_observed(state) >= BUDGET:
        print(json.dumps({{"error": f"evaluation budget exhausted ({{BUDGET}})"}}))
        sys.exit(1)

    out = subprocess.run(
        [INTERPRETER, str(IMPL), sys.argv[1]], capture_output=True, text=True
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr or out.stdout or "oracle failed\n")
        sys.exit(out.returncode)
    try:
        metrics = json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.stdout.write(out.stdout)
        sys.exit(0)

    trial = {{
        "trial_id": str(uuid.uuid4())[:8],
        "config": config,
        "metrics": metrics,
        "status": "observed",
        "created_at": time.time(),
        "observed_at": time.time(),
        "x_gp": None,
    }}
    state.setdefault("trials", []).append(trial)
    STATE.write_text(json.dumps(state, indent=2, default=str))
    sys.stdout.write(out.stdout if out.stdout.endswith("\n") else out.stdout + "\n")


if __name__ == "__main__":
    main()
'''


def _scrub_context(text: str) -> str:
    """Remove lenz instructions from generated context.md without naming a substitute library."""
    replacements = (
        (
            "Use only `lenz` and `./oracle` to search; every evaluation must go through `./oracle`.",
            "Propose configurations yourself and evaluate them with `./oracle`; every evaluation must go through `./oracle`.",
        ),
        (
            "Use only `lenz` and `./oracle` to search; ",
            "Propose configurations yourself and evaluate them with `./oracle`. ",
        ),
    )
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    # belt: any leftover mention (split strings, future templates) is stripped
    out = re.sub(r"`lenz`", "your own proposals", out, flags=re.IGNORECASE)
    out = re.sub(r"\blenz\b", "an external optimizer", out, flags=re.IGNORECASE)
    return out


def install_sara_only(
    sandbox: Path,
    space: dict,
    objectives: dict,
    constraints: list[dict] | None,
    budget: int,
    preserve_trials: bool = False,
) -> None:
    """Move the real oracle aside, wrap it, and write a lenz-free state.json.

    `preserve_trials`: keep whatever `trials` are already in `state.json`
    instead of resetting to `[]`. Set this when the caller pre-seeded a real
    Sobol warm-start (via `lenz`, run by the harness itself -- never exposed
    to the agent) before switching the sandbox into no-optimizer mode, so
    sara-only starts from the same kind of head start its lenz-backed
    sibling conditions get instead of always opening at evaluation #1.
    """
    sandbox = Path(sandbox)
    oracle = sandbox / "oracle"
    impl = sandbox / ".oracle_impl"
    if not oracle.exists() and not impl.exists():
        raise FileNotFoundError(f"no oracle in {sandbox}")
    if oracle.exists() and not impl.exists():
        shutil.move(str(oracle), str(impl))
        if os.name != "nt":
            impl.chmod(impl.stat().st_mode | 0o111)

    (sandbox / "oracle").write_text(_WRAPPER.format(budget=int(budget), interpreter=sys.executable))
    if os.name != "nt":
        (sandbox / "oracle").chmod((sandbox / "oracle").stat().st_mode | 0o111)

    existing_trials: list = []
    state_path = sandbox / "state.json"
    if preserve_trials and state_path.is_file():
        try:
            existing_trials = json.loads(state_path.read_text()).get("trials", [])
        except json.JSONDecodeError:
            existing_trials = []

    obj_list = [{"metric": m, "minimize": (d == "minimize")} for m, d in objectives.items()]
    state = {
        "space": space,
        "shelf": {
            "objectives": obj_list,
            "constraints": constraints or [],
            "acqf": "none",
            "acqf_params": {},
            "bounds": {},
            "surrogate": "none",
        },
        "trials": existing_trials,
        "events": [],
        "pending_x_gp": [],
    }
    state_path.write_text(json.dumps(state, indent=2, default=str))

    ctx = sandbox / "context.md"
    if ctx.is_file():
        ctx.write_text(_scrub_context(ctx.read_text()))
