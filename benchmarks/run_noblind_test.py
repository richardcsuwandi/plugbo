"""Run `sara` against a *no-blind* benchmark sandbox: the benchmark's real
identity, real parameter names, and (by default) its real, unshifted textbook
bounds are all disclosed in context.md -- the exact opposite of
`run_blind_test.py`'s anti-memorization transform.

The paper observes that Sara can often *recognize* a shifted, renamed
benchmark mid-run from the pattern of its own evaluations ("Benchmark
recognition and evaluation") -- but because the optimum was
shifted, recognition alone couldn't let it one-shot the answer. This script
asks a narrower, more direct question: with the shift removed and the name
handed over up front, how much of a model's "optimization" performance on a
named textbook function is actually optimum recall? A model that has truly
memorized e.g. Hartmann6's optimum should be able to submit it -- or
something very close -- as its very first evaluation.

Two conditions, both built by `benchmarks.sandbox.build_sandbox(reveal=True, ...)`:
  - `shift=False` (default): pure recall probe. The optimum sits exactly
    where the textbook says it does.
  - `shift=True`: the model is told what it's solving but the optimum has
    still been relocated (same per-seed shift as the blind condition) -- it
    knows the map but not the treasure's exact square.

Unlike `run_blind_test.py`, warm-start defaults to 0: any pre-seeded Sobol
evaluations would land before the agent's own first move and defeat the
point of measuring evaluation #1.

Usage:
    python3 -m benchmarks.run_noblind_test \\
        --benchmark hartmann6 --provider anthropic --model claude-opus-5 \\
        --budget 5 --seed 42 --root ./results/logs/noblind-hartmann6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sara.agent import run_campaign
from sara.cli import _now_iso, _system_prompt, _system_prompt_sara_only, _user_prompt, _user_prompt_sara_only, _write_meta
from llm.factory import get_client
from lenz.llm_config import export_api_key, stamp_workdir

from .lenz_loop import create_and_warmup
from .obfuscate import ObfuscatedBenchmark
from .run_blind_test import _backend_directive, _lenz_create_args, score_sandbox
from .sandbox import build_sandbox


def one_shot_summary(scored: dict, tol: float) -> dict:
    """Whether the first observed trial already landed within `tol` of the
    true optimum (feasible, for constrained benchmarks) -- the headline
    number for "did it one-shot from memorization."
    """
    trials = scored["trials"]
    if not trials:
        return {"first_trial_regret": None, "one_shot_success": False, "tol": tol}
    first = trials[0]
    feasible = first.get("feasible", True)
    regret = first["regret"]
    return {
        "first_trial_regret": regret,
        "first_trial_feasible": feasible,
        "one_shot_success": feasible and abs(regret) <= tol,
        "tol": tol,
    }


def run_noblind_test(
    benchmark_name: str,
    provider: str,
    model: str,
    budget: int,
    root: Path,
    seed: int | None = None,
    shift: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    surrogate: str = "fixed",
    acqf: str = "noisy_logei",
    extra_body: str | None = None,
    warmup: int = 0,
    one_shot_tol: float = 1e-2,
    kernel_llm_provider: str | None = None,
    kernel_llm_model: str | None = None,
    kernel_llm_base_url: str | None = None,
    kernel_llm_api_key_env: str | None = None,
    kernel_llm_extra_body: str | None = None,
    no_lenz: bool = False,
    context_variant: str = "domain",
) -> dict:
    built = build_sandbox(
        benchmark_name,
        root=root,
        seed=seed,
        reveal=True,
        shift=shift,
        context_variant=context_variant,
    )
    sandbox: Path = built["sandbox"]
    export_api_key(provider, api_key)

    secret = json.loads(built["secret_path"].read_text())
    ob = ObfuscatedBenchmark.from_secret(secret)

    if no_lenz:
        from .sara_only import install_sara_only

        # Opt-in only (see run_blind_test.py's identical comment): a
        # --no-lenz run that never passes --warmup keeps its historical
        # zero-warmup behavior. Passing --warmup N runs the same real
        # Sobol warm-start the lenz-backed sibling conditions get, via the
        # harness itself (never exposed to the agent), so the comparison
        # isn't silently handing sara-only N free extra evaluations.
        n_warm = max(0, warmup) if warmup else 0
        already = 0
        if n_warm:
            warm_create_args = _lenz_create_args(
                surrogate, acqf, budget, seed=seed, constraints=built["constraints"]
            )
            already = create_and_warmup(
                sandbox, ob.unit_space_json(), warm_create_args, n_warm, budget, objectives=ob.objectives_json()
            )

        install_sara_only(
            sandbox,
            space=ob.unit_space_json(),
            objectives=ob.objectives_json(),
            constraints=built["constraints"],
            budget=budget,
            preserve_trials=already > 0,
        )
        user_prompt = _user_prompt_sara_only((sandbox / "context.md").read_text(), "./oracle", budget)
        if already:
            user_prompt += (
                f"\n\n{already} evaluation(s) already ran before you started (uninformed "
                f"space-filling, not chosen by you) and are logged in `./state.json`. "
                f"{budget - already} evaluation(s) remain in your budget.\n"
            )
        system_prompt = _system_prompt_sara_only()
    else:
        create_args = _lenz_create_args(
            surrogate,
            acqf,
            budget,
            seed=seed,
            constraints=built["constraints"],
            llm_provider=provider,
            llm_model=model,
            llm_base_url=base_url,
            llm_extra_body=extra_body,
            llm_api_key_env=None if (kernel_llm_provider and kernel_llm_model) else kernel_llm_api_key_env,
            kernel_llm_provider=kernel_llm_provider,
            kernel_llm_model=kernel_llm_model,
            kernel_llm_base_url=kernel_llm_base_url,
            kernel_llm_api_key_env=kernel_llm_api_key_env,
            kernel_llm_extra_body=kernel_llm_extra_body,
        )
        n_warm = max(0, warmup)
        already = create_and_warmup(
            sandbox, ob.unit_space_json(), create_args, n_warm, budget, objectives=ob.objectives_json()
        )
        cake_override = {}
        if kernel_llm_provider and kernel_llm_model:
            cake_override = {
                "cake": {
                    "provider": kernel_llm_provider,
                    "model": kernel_llm_model,
                    "base_url": kernel_llm_base_url,
                    "api_key_env": kernel_llm_api_key_env,
                    "extra_body": kernel_llm_extra_body,
                }
            }
        stamp_workdir(
            sandbox,
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "extra_body": extra_body,
            },
            cake_override,
        )
        user_prompt = _user_prompt((sandbox / "context.md").read_text(), "./oracle", budget)
        user_prompt += _backend_directive(already, budget, create_args)
        system_prompt = _system_prompt()

    parsed_extra_body = json.loads(extra_body) if extra_body else None
    client = get_client(provider, model, base_url=base_url, api_key=api_key, extra_body=parsed_extra_body)

    trace_path = sandbox / "trace.jsonl"
    meta_path = sandbox / "run_meta.json"
    meta = {
        "kind": "sara-only" if no_lenz else "sara-noblind",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "eval_cmd": "./oracle",
        "budget": budget,
        "started_at": _now_iso(),
        "ended_at": None,
        "status": "running",
        "surrogate": "none" if no_lenz else surrogate,
        "acqf": "none" if no_lenz else acqf,
        "seed": seed,
        "warmup": already,
        "reveal": True,
        "shift": shift,
        "no_lenz": no_lenz,
        "context_variant": context_variant,
    }
    _write_meta(meta_path, meta)

    try:
        result = run_campaign(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sandbox=sandbox,
            trace_path=trace_path,
            budget=budget,
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

    scored = score_sandbox(built, sandbox)
    return {
        **scored,
        **one_shot_summary(scored, one_shot_tol),
        "shifted": shift,
        "final_message": result.final_message,
        "n_agent_steps": result.n_steps,
        "trace_path": str(trace_path),
    }


def main() -> None:
    from .functions import REGISTRY

    p = argparse.ArgumentParser(
        description="Run sara against a no-blind (identity-revealed) benchmark sandbox -- tests one-shot memorization."
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help=f"one of {sorted(REGISTRY)}, or 'gp_sample<dim>' (a meaningless condition here -- there's no name to recall)",
    )
    p.add_argument("--provider", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--budget", type=int, default=5, help="small by design -- this probes evaluation #1, not a full campaign")
    p.add_argument("--root", default="./runs/noblind")
    p.add_argument("--seed", type=int, default=None, help="pins the Sobol warm-start and (with --shift) the relocation")
    p.add_argument(
        "--shift",
        action="store_true",
        help="also relocate the optimum (same transform as the blind condition) -- tests adaptation from a revealed prior, not pure recall",
    )
    p.add_argument("--warmup", type=int, default=0, help="Sobol evals before sara starts (default 0 -- keep evaluation #1 the agent's own choice)")
    p.add_argument("--surrogate", default="fixed", choices=["fixed", "cake"])
    p.add_argument("--acqf", default="noisy_logei")
    p.add_argument("--kernel-llm-provider", default=None, help="CAKE override; defaults to --provider/--model")
    p.add_argument("--kernel-llm-model", default=None, help="CAKE override; defaults to --provider/--model")
    p.add_argument("--kernel-llm-base-url", default=None, help="required if --kernel-llm-provider is openai-compatible")
    p.add_argument("--kernel-llm-api-key-env", default=None, help="name of the env var holding the kernel LLM's key -- never the key itself")
    p.add_argument("--kernel-llm-extra-body", default=None, help="JSON object merged into the kernel LLM's request body, e.g. '{\"enable_thinking\": false}'")
    p.add_argument("--extra-body", default=None, help="JSON object merged into sara's own LLM request body")
    p.add_argument("--one-shot-tol", type=float, default=1e-2, help="absolute regret threshold counted as a 'hit' on evaluation #1")
    p.add_argument(
        "--no-lenz",
        action="store_true",
        help="pure LLM optimizer: Sara proposes every point; lenz is not created and cannot be called",
    )
    p.add_argument(
        "--context-variant",
        default="domain",
        choices=["domain", "generic", "misleading"],
        help="bolt_lora only: domain (LoRA/Qwen story), generic (names/types, no domain prose), "
        "misleading (false LoRA folklore). Other benchmarks must stay at domain.",
    )
    args = p.parse_args()

    result = run_noblind_test(
        benchmark_name=args.benchmark,
        provider=args.provider,
        model=args.model,
        budget=args.budget,
        root=Path(args.root),
        seed=args.seed,
        shift=args.shift,
        base_url=args.base_url,
        api_key=args.api_key,
        surrogate=args.surrogate,
        acqf=args.acqf,
        extra_body=args.extra_body,
        warmup=args.warmup,
        one_shot_tol=args.one_shot_tol,
        kernel_llm_provider=args.kernel_llm_provider,
        kernel_llm_model=args.kernel_llm_model,
        kernel_llm_base_url=args.kernel_llm_base_url,
        kernel_llm_api_key_env=args.kernel_llm_api_key_env,
        kernel_llm_extra_body=args.kernel_llm_extra_body,
        no_lenz=args.no_lenz,
        context_variant=args.context_variant,
    )

    print("\n=== No-blind (identity-revealed) test result ===")
    print(f"Benchmark: {result['benchmark']} (revealed to the agent; shift={'on' if result['shifted'] else 'off'})")
    print(f"Evaluations observed: {result['n_observed']}")
    if result["first_trial_regret"] is not None:
        print(f"Evaluation #1 regret: {result['first_trial_regret']:.6f}  (tol {result['tol']:.4f})")
    print(f"One-shot success: {result['one_shot_success']}")
    print(f"Best true regret: {result['best_regret']:.6f}  (true f_opt = {result['true_f_opt']:.6f})")
    print(f"Sandbox: {result['sandbox']}")
    print(f"\nAgent's final report:\n{result['final_message']}")


if __name__ == "__main__":
    main()
