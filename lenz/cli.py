"""The lenz command-line interface. Every invocation loads `--state`, runs
one command, saves the result, and prints exactly one JSON line.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import commands as C
from .acquisition import AcqfError
from .models import ModelError
from .optimize import OptimizeError
from .space import SpaceError
from .state import Frame, StateError

load_dotenv()


def sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def _print_ok(command: str, result) -> None:
    print(json.dumps({"ok": True, "command": command, "result": sanitize(result)}, default=str))


def _print_err(command: str, message: str, **extra) -> None:
    payload = {"ok": False, "command": command, "error": message}
    payload.update(extra)
    print(json.dumps(payload, default=str))


def build_parser() -> argparse.ArgumentParser:
    # `--state` is accepted by every subcommand (e.g. `lenz create --state ./state.json ...`),
    # not as a global flag before the subcommand -- that's the calling convention documented
    # in LENZ_REF.md and used throughout the paper's examples.
    state_parent = argparse.ArgumentParser(add_help=False)
    state_parent.add_argument("--state", required=True, help="path to state.json")

    p = argparse.ArgumentParser(prog="lenz")
    sub = p.add_subparsers(dest="command", required=True)

    def _add_kernel_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--budget", type=int, default=None, help="total evaluation budget (used by the kernel-evolution freeze schedule)")
        sp.add_argument("--kernel-llm-provider", default=None)
        sp.add_argument("--kernel-llm-model", default=None)
        sp.add_argument("--kernel-llm-base-url", default=None)
        sp.add_argument("--kernel-llm-api-key-env", default=None, help="env var NAME holding the key -- never the key itself")
        sp.add_argument("--kernel-llm-extra-body", default=None, help="JSON object merged into the kernel LLM's request body, e.g. '{\"enable_thinking\": false}'")
        sp.add_argument("--kernel-population-size", type=int, default=None)
        sp.add_argument("--kernel-init-after", type=int, default=None)
        sp.add_argument("--kernel-evolve-every", type=int, default=None)
        sp.add_argument("--kernel-freeze-fraction", type=float, default=None)
        sp.add_argument("--kernel-num-crossover", type=int, default=None)
        sp.add_argument("--kernel-mutation-prob", type=float, default=None)

    c = sub.add_parser("create", parents=[state_parent])
    c.add_argument("--space", required=True)
    c.add_argument("--objectives", required=True)
    c.add_argument("--constraints", default=None)
    c.add_argument("--acqf", default="noisy_logei")
    c.add_argument("--beta", type=float, default=None)
    c.add_argument("--surrogate", default="fixed", choices=["fixed", "cake"])
    c.add_argument("--seed", type=int, default=None, help="pins Sobol warmup (and --acqf sobol) so the same seed replays the same initial design")
    c.add_argument("--force", action="store_true", help="overwrite an existing state.json")
    _add_kernel_args(c)

    s = sub.add_parser("suggest", parents=[state_parent])
    s.add_argument("--q", type=int, default=1)
    s.add_argument("--bounds", default=None)
    s.add_argument("--around", nargs="?", const=True, default=None)
    s.add_argument("--radius", type=float, default=0.1)

    sb = sub.add_parser("submit", parents=[state_parent])
    sb.add_argument("--config", required=True)
    sb.add_argument("--metrics", default=None)

    ob = sub.add_parser("observe", parents=[state_parent])
    ob.add_argument("--config", required=True)
    ob.add_argument("--metrics", required=True)

    sbnd = sub.add_parser("set-bounds", parents=[state_parent])
    sbnd.add_argument("--bounds", required=True)

    sacqf = sub.add_parser("set-acqf", parents=[state_parent])
    sacqf.add_argument("--acqf", required=True)
    sacqf.add_argument("--beta", type=float, default=None)

    sobj = sub.add_parser("set-objectives", parents=[state_parent])
    sobj.add_argument("--objectives", required=True)

    scon = sub.add_parser("set-constraints", parents=[state_parent])
    scon.add_argument("--constraints", required=True)

    ssur = sub.add_parser("set-surrogate", parents=[state_parent])
    ssur.add_argument("--surrogate", required=True, choices=["fixed", "cake"])
    _add_kernel_args(ssur)

    evk = sub.add_parser("evolve-kernels", parents=[state_parent])
    evk.add_argument("--force", action="store_true")

    sub.add_parser("kernel-population", parents=[state_parent])

    sub.add_parser("status", parents=[state_parent])
    sub.add_parser("diagnostics", parents=[state_parent])

    pr = sub.add_parser("predict", parents=[state_parent])
    pr.add_argument("--configs", required=True)

    sc = sub.add_parser("score", parents=[state_parent])
    sc.add_argument("--configs", required=True)
    sc.add_argument("--acqf", required=True)

    sub.add_parser("trials", parents=[state_parent])

    inc = sub.add_parser("incumbent", parents=[state_parent])
    inc.add_argument("--in-bounds", action="store_true")

    sub.add_parser("pareto", parents=[state_parent])

    return p


DISPATCH = {
    "suggest": C.cmd_suggest,
    "submit": C.cmd_submit,
    "observe": C.cmd_observe,
    "set-bounds": C.cmd_set_bounds,
    "set-acqf": C.cmd_set_acqf,
    "set-objectives": C.cmd_set_objectives,
    "set-constraints": C.cmd_set_constraints,
    "set-surrogate": C.cmd_set_surrogate,
    "evolve-kernels": C.cmd_evolve_kernels,
    "kernel-population": C.cmd_kernel_population,
    "status": C.cmd_status,
    "diagnostics": C.cmd_diagnostics,
    "predict": C.cmd_predict,
    "score": C.cmd_score,
    "trials": C.cmd_trials,
    "incumbent": C.cmd_incumbent,
    "pareto": C.cmd_pareto,
}

KNOWN_ERRORS = (SpaceError, StateError, ModelError, OptimizeError, AcqfError, KeyError, ValueError, TypeError)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    try:
        if command == "create":
            if Path(args.state).exists() and not args.force:
                _print_err(command, f"state already exists at '{args.state}'; pass --force to overwrite")
                return
            frame, result = C.cmd_create(args)
            frame.save(args.state)
            _print_ok(command, result)
            return

        try:
            frame = Frame.load(args.state)
        except FileNotFoundError:
            _print_err(command, f"no state found at '{args.state}'; run 'create' first")
            return

        fn = DISPATCH[command]
        new_frame, result = fn(frame, args)
        if new_frame is not None:
            new_frame.save(args.state)
        _print_ok(command, result)

    except C.NoMatchingSubmission as e:
        _print_err(command, str(e), outstanding=e.outstanding)
    except json.JSONDecodeError as e:
        _print_err(command, f"malformed JSON: {e}")
    except KNOWN_ERRORS as e:
        _print_err(command, str(e))


if __name__ == "__main__":
    main(sys.argv[1:])
