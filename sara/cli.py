"""`sara run` -- launch an agent-driven Bayesian optimization campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_campaign
from llm.factory import PROVIDERS, get_client

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _system_prompt() -> str:
    system = (PROMPTS_DIR / "SYSTEM.md").read_text()
    lenz_ref = (PROMPTS_DIR / "LENZ_REF.md").read_text()
    return f"{system}\n\n{lenz_ref}"


def _system_prompt_sara_only() -> str:
    return (PROMPTS_DIR / "SARA_ONLY.md").read_text()


def _user_prompt(context_text: str, eval_cmd: str, budget: int) -> str:
    return (
        f"{context_text.strip()}\n\n"
        f"Run the experiment as: {eval_cmd} '<config-json>' -- it prints the metrics as a JSON object.\n"
        f"Budget: {budget} evaluations.\n"
        "Keep your lenz state in './state.json' (relative to your sandbox working directory).\n"
        "Treat the experiment as a black box: do not open or read its implementation; measure it only by running it."
    )


def _user_prompt_sara_only(context_text: str, eval_cmd: str, budget: int) -> str:
    return (
        f"{context_text.strip()}\n\n"
        f"Run the experiment as: {eval_cmd} '<config-json>' -- it prints the metrics as a JSON object.\n"
        f"Budget: {budget} evaluations.\n"
        "You propose every configuration yourself. Successful oracle calls are recorded automatically.\n"
        "Treat the experiment as a black box: do not open or read its implementation; measure it only by running it."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sara")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run an agent-driven optimization campaign")
    r.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", default=None, help="required for --provider openai-compatible")
    r.add_argument("--api-key", default=None, help="defaults to the provider's standard env var")
    r.add_argument("--context", required=True, help="path to a natural-language problem description")
    r.add_argument(
        "--eval",
        required=True,
        help="shell command Sara runs to evaluate a config, e.g. 'python3 eval.py' (config JSON is appended)",
    )
    r.add_argument("--budget", type=int, required=True, help="target number of real evaluations")
    r.add_argument("--workdir", required=True, help="sandbox directory for lenz state + evaluation script")
    r.add_argument("--trace", default=None, help="path to write a JSONL deliberation trace (default: <workdir>/trace.jsonl)")
    r.add_argument(
        "--no-lenz",
        action="store_true",
        help="pure LLM optimizer: no lenz CLI, no GP, no acquisition. Sara proposes every point.",
    )

    return p


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    context_text = Path(args.context).read_text()
    no_lenz = bool(getattr(args, "no_lenz", False))
    if no_lenz:
        system_prompt = _system_prompt_sara_only()
        user_prompt = _user_prompt_sara_only(context_text, args.eval, args.budget)
    else:
        system_prompt = _system_prompt()
        user_prompt = _user_prompt(context_text, args.eval, args.budget)

    client = get_client(args.provider, args.model, base_url=args.base_url, api_key=args.api_key)
    trace_path = Path(args.trace) if args.trace else workdir / "trace.jsonl"
    meta_path = workdir / "run_meta.json"

    meta = {
        "kind": "sara-only" if no_lenz else "sara",
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "context_path": str(Path(args.context).resolve()),
        "eval_cmd": args.eval,
        "budget": args.budget,
        "started_at": _now_iso(),
        "ended_at": None,
        "status": "running",
        "no_lenz": no_lenz,
    }
    _write_meta(meta_path, meta)

    try:
        result = run_campaign(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sandbox=workdir,
            trace_path=trace_path,
            budget=args.budget,
            block_lenz=no_lenz,
        )
    except Exception as e:
        meta.update(ended_at=_now_iso(), status="failed", error=str(e))
        _write_meta(meta_path, meta)
        raise

    meta.update(
        ended_at=_now_iso(),
        status="completed",
        n_steps=result.n_steps,
        final_message=result.final_message,
        usage=result.usage_total,
    )
    _write_meta(meta_path, meta)

    print(result.final_message)
    print(f"\n[{result.n_steps} agent steps; trace written to {trace_path}]")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
