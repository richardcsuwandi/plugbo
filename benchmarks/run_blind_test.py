"""Run `sara` against a blind (anti-memorization) benchmark sandbox and score
true regret afterward, using the secret this script has access to but the
agent never did. This is the harness for reproducing the paper's "what
happens with no exploitable information" condition ("On the experimental
setup").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sara.agent import run_campaign
from sara.cli import _now_iso, _system_prompt, _user_prompt, _write_meta
from llm.factory import get_client

from .lenz_loop import create_and_warmup, warmup_n
from .obfuscate import ObfuscatedBenchmark
from .sandbox import build_sandbox


def score_sandbox(built: dict, sandbox: Path) -> dict:
    """Scores whatever trials landed in `sandbox/state.json` against the true
    (never-exposed) benchmark identity recorded in `built["secret_path"]`.
    Shared by both the sara-driven runner below and the non-agentic
    baseline runner (`run_blind_baseline.py`), so the two are scored
    identically.

    For constrained benchmarks (a "c" metric present), `best_regret` is the
    best regret among *feasible* trials only (c <= constraint_upper) --
    infinite if none are feasible yet -- since an infeasible y isn't a
    meaningful answer to "how close did it get." Each trial still reports its
    own `feasible` flag for full transparency.
    """
    secret = json.loads(built["secret_path"].read_text())
    ob = ObfuscatedBenchmark.from_secret(secret)
    spec = ob.spec
    constrained = spec.constraint_fn is not None

    trials = []
    best_regret = float("inf")
    state_path = sandbox / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for t in state.get("trials", []):
            metrics = t.get("metrics") or {}
            if t.get("status") != "observed" or "y" not in metrics:
                continue
            y = float(metrics["y"])
            regret = y - spec.f_opt
            trial = {"config": t["config"], "y": y, "regret": regret}
            feasible = True
            if constrained:
                feasible = "c" in metrics and float(metrics["c"]) <= spec.constraint_upper
                trial["c"] = metrics.get("c")
                trial["feasible"] = feasible
            if feasible:
                best_regret = min(best_regret, regret)
            trials.append(trial)

    return {
        "benchmark": spec.name,
        "true_f_opt": spec.f_opt,
        "n_observed": len(trials),
        "best_regret": best_regret,
        "trials": trials,
        "sandbox": str(sandbox),
    }


def _lenz_create_args(
    surrogate: str,
    acqf: str,
    seed: int | None = None,
    kernel_llm_provider: str | None = None,
    kernel_llm_model: str | None = None,
    kernel_llm_base_url: str | None = None,
    kernel_llm_api_key_env: str | None = None,
    kernel_llm_extra_body: str | None = None,
    constraints: list[dict] | None = None,
) -> list[str]:
    if constraints and surrogate == "cake":
        raise ValueError(
            "this benchmark is constrained, and surrogate 'cake' only supports single-objective, "
            "unconstrained studies -- pass --surrogate fixed instead"
        )
    args = ["--acqf", acqf, "--surrogate", surrogate]
    if seed is not None:
        args += ["--seed", str(seed)]
    if constraints:
        args += ["--constraints", json.dumps(constraints)]
    if surrogate == "cake":
        if not (kernel_llm_provider and kernel_llm_model):
            raise ValueError("--surrogate cake requires --kernel-llm-provider and --kernel-llm-model")
        args += ["--kernel-llm-provider", kernel_llm_provider, "--kernel-llm-model", kernel_llm_model]
        if kernel_llm_base_url:
            args += ["--kernel-llm-base-url", kernel_llm_base_url]
        if kernel_llm_api_key_env:
            args += ["--kernel-llm-api-key-env", kernel_llm_api_key_env]
        if kernel_llm_extra_body:
            args += ["--kernel-llm-extra-body", kernel_llm_extra_body]
    return args


def _backend_directive(n_warmup: int, budget: int, create_args: list[str]) -> str:
    """Pins the backend configuration sara would otherwise choose herself,
    so `--surrogate`/`--acqf` are controlled experimental conditions rather
    than left to the agent's judgment (e.g. for a vanilla-vs-cake comparison
    at fixed acquisition function).
    """
    flags = " ".join(create_args)
    if n_warmup > 0:
        return (
            "\n\nBackend configuration (fixed for this experiment -- follow exactly, do not deviate): "
            f"a lenz study already exists at `./state.json` with {n_warmup} Sobol warm-start "
            f"evaluations already recorded (they count toward the budget of {budget}). "
            "Do not call `lenz create` -- that would wipe the shared initial design. "
            "Continue from the existing state. "
            "Do not call `set-acqf` or `set-surrogate` at any point during the run -- keep both fixed "
            "for the entire campaign, even if you'd otherwise want to adapt them."
        )
    return (
        "\n\nBackend configuration (fixed for this experiment -- follow exactly, do not deviate): "
        f"when you call `lenz create`, pass `{flags}`. "
        "Do not call `set-acqf` or `set-surrogate` at any point during the run -- keep both fixed "
        "for the entire campaign, even if you'd otherwise want to adapt them."
    )


def run_blind_test(
    benchmark_name: str,
    provider: str,
    model: str,
    budget: int,
    root: Path,
    seed: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    surrogate: str = "fixed",
    acqf: str = "noisy_logei",
    kernel_llm_provider: str | None = None,
    kernel_llm_model: str | None = None,
    kernel_llm_base_url: str | None = None,
    kernel_llm_api_key_env: str | None = None,
    kernel_llm_extra_body: str | None = None,
    extra_body: str | None = None,
    warmup: int | None = None,
) -> dict:
    built = build_sandbox(benchmark_name, root=root, seed=seed)
    sandbox: Path = built["sandbox"]

    secret = json.loads(built["secret_path"].read_text())
    ob = ObfuscatedBenchmark.from_secret(secret)
    create_args = _lenz_create_args(
        surrogate,
        acqf,
        seed=seed,
        kernel_llm_provider=kernel_llm_provider,
        kernel_llm_model=kernel_llm_model,
        kernel_llm_base_url=kernel_llm_base_url,
        kernel_llm_api_key_env=kernel_llm_api_key_env,
        kernel_llm_extra_body=kernel_llm_extra_body,
        constraints=built["constraints"],
    )
    n_warm = warmup_n(len(ob.param_names), seed, warmup)
    already = create_and_warmup(sandbox, ob.unit_space_json(), create_args, n_warm, budget)

    parsed_extra_body = json.loads(extra_body) if extra_body else None
    client = get_client(provider, model, base_url=base_url, api_key=api_key, extra_body=parsed_extra_body)
    context_text = (sandbox / "context.md").read_text()
    user_prompt = _user_prompt(context_text, "./oracle", budget)
    user_prompt += _backend_directive(already, budget, create_args)

    trace_path = sandbox / "trace.jsonl"
    meta_path = sandbox / "run_meta.json"
    meta = {
        "kind": "sara",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "eval_cmd": "./oracle",
        "budget": budget,
        "started_at": _now_iso(),
        "ended_at": None,
        "status": "running",
        "surrogate": surrogate,
        "acqf": acqf,
        "seed": seed,
        "warmup": already,
    }
    _write_meta(meta_path, meta)

    try:
        result = run_campaign(
            client=client,
            system_prompt=_system_prompt(),
            user_prompt=user_prompt,
            sandbox=sandbox,
            trace_path=trace_path,
            budget=budget,
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

    scored = score_sandbox(built, sandbox)
    return {
        **scored,
        "final_message": result.final_message,
        "n_agent_steps": result.n_steps,
        "trace_path": str(trace_path),
    }


def main() -> None:
    from .functions import REGISTRY

    p = argparse.ArgumentParser(description="Run sara against a blind (anti-memorization) benchmark sandbox.")
    p.add_argument(
        "--benchmark",
        required=True,
        help=f"one of {sorted(REGISTRY)}, or 'gp_sample<dim>' (e.g. 'gp_sample6') for a fresh random GP sample path",
    )
    p.add_argument("--provider", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--root", default="./runs/blind")
    p.add_argument("--seed", type=int, default=None, help="pins the renaming/shift transform AND the Sobol warm-start")
    p.add_argument("--warmup", type=int, default=None, help="shared Sobol evaluations before sara starts (default: d+1 when --seed is set, else 0)")
    p.add_argument("--surrogate", default="fixed", choices=["fixed", "cake"], help="pinned for the whole run; sara is instructed not to change it")
    p.add_argument("--acqf", default="noisy_logei", help="pinned for the whole run; sara is instructed not to change it")
    p.add_argument("--kernel-llm-provider", default=None, help="required if --surrogate cake")
    p.add_argument("--kernel-llm-model", default=None, help="required if --surrogate cake")
    p.add_argument("--kernel-llm-base-url", default=None, help="required if --kernel-llm-provider is openai-compatible")
    p.add_argument("--kernel-llm-api-key-env", default=None, help="name of the env var holding the kernel LLM's key -- never the key itself")
    p.add_argument("--kernel-llm-extra-body", default=None, help="JSON object merged into the kernel LLM's request body, e.g. '{\"enable_thinking\": false}'")
    p.add_argument("--extra-body", default=None, help="JSON object merged into sara's own LLM request body, e.g. '{\"enable_thinking\": false}'")
    args = p.parse_args()

    result = run_blind_test(
        benchmark_name=args.benchmark,
        provider=args.provider,
        model=args.model,
        budget=args.budget,
        root=Path(args.root),
        seed=args.seed,
        base_url=args.base_url,
        api_key=args.api_key,
        surrogate=args.surrogate,
        acqf=args.acqf,
        kernel_llm_provider=args.kernel_llm_provider,
        kernel_llm_model=args.kernel_llm_model,
        kernel_llm_base_url=args.kernel_llm_base_url,
        kernel_llm_api_key_env=args.kernel_llm_api_key_env,
        kernel_llm_extra_body=args.kernel_llm_extra_body,
        extra_body=args.extra_body,
        warmup=args.warmup,
    )

    print("\n=== Blind test result ===")
    print(f"True benchmark: {result['benchmark']} (revealed post-hoc; the agent never saw this)")
    print(f"Evaluations observed: {result['n_observed']}")
    print(f"Best true regret: {result['best_regret']:.6f}  (true f_opt = {result['true_f_opt']:.6f})")
    print(f"Sandbox: {result['sandbox']}")
    print(f"\nAgent's final report:\n{result['final_message']}")


if __name__ == "__main__":
    main()
