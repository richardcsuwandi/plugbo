"""Non-agentic baselines for the blind test: drive `lenz` directly through a
plain scripted suggest/submit/observe loop -- no LLM, no Sara, no mid-run
reconfiguration -- against the same blind sandbox oracle used by
`run_blind_test.py`. The "vanilla" policy below is the paper's "vanilla BO"
condition ("without explicit ownership, the agent can ... reduc[e]
the loop to vanilla BO"): a fixed surrogate and a fixed acquisition function
for the whole run, scored the same way as the sara-driven runs for a
like-for-like comparison.

To add another baseline/config, add an entry to POLICIES below -- nothing
else needs to change. `lenz_create_args` is appended verbatim to `lenz
create` (after `--space`/`--objectives`), so anything `lenz create` accepts
(acqf, surrogate, kernel-llm-*, ...) works here. For a policy whose candidate
selection isn't just "call `lenz suggest`" (e.g. pure random sampling
bypassing lenz's own Sobol engine), pass a custom `suggest` callable instead.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from .lenz_loop import create_and_warmup, evaluate, lenz, warmup_n
from .obfuscate import ObfuscatedBenchmark
from .priors import get_prior_fixture
from .run_blind_test import score_sandbox
from .sandbox import build_sandbox


@dataclass
class Policy:
    """Backward-compatible named configuration for the four plugin slots."""

    lenz_create_args: list[str] = field(default_factory=list)
    surrogate: str = "fixed"
    region: str = "box"
    sampler: str = "botorch"


# Add new baselines/configs here -- name -> Policy. Keep names CLI-friendly
# (used as `--policy <name>`); they also become the run's log-dir suffix.
POLICIES: dict[str, Policy] = {
    "vanilla": Policy(lenz_create_args=["--acqf", "noisy_logei"]),
    "sobol": Policy(lenz_create_args=["--acqf", "sobol"]),
    # Kernel evolution without an agent driving BO decisions -- still needs
    # --kernel-llm-provider/--kernel-llm-model on the CLI (cake's kernel
    # evolution makes its own small LLM calls independent of "no sara" here).
    "cake": Policy(lenz_create_args=["--acqf", "noisy_logei"], surrogate="cake"),
    "turbo": Policy(lenz_create_args=["--acqf", "noisy_logei"], region="turbo"),
}


def _configure_slots(
    state_path: Path,
    *,
    region: str,
    sampler: str,
    prior: dict | None,
    decay_beta: float,
) -> None:
    if region == "turbo":
        lenz(state_path, "set-region", "--policy", "turbo")
    if sampler == "llambo":
        lenz(state_path, "set-sampler", "--sampler", "llambo")
    if prior:
        lenz(
            state_path,
            "set-belief",
            "--prior",
            json.dumps(prior),
            "--decay-beta",
            str(decay_beta),
        )


def run_blind_baseline(
    benchmark_name: str,
    budget: int,
    root: Path,
    policy_name: str = "vanilla",
    seed: int | None = None,
    extra_create_args: list[str] | None = None,
    warmup: int | None = None,
    reveal: bool = False,
    shift: bool = False,
    context_variant: str = "domain",
    surrogate: str | None = None,
    region: str | None = None,
    sampler: str | None = None,
    prior: dict | None = None,
    prior_fixture: str | None = None,
    decay_beta: float = 10.0,
) -> dict:
    """`reveal`/`shift` exist for symmetry with `run_blind_test.py`/
    `run_noblind_test.py` -- a non-agentic policy never reads `context.md`,
    so revealing identity can't change its *behavior*. What `shift` still
    changes is the actual evaluated function (the oracle really does move
    the optimum), which makes an unshifted, revealed-bounds vanilla run a
    useful "floor": how fast would a memorization-free method solve this if
    the optimum weren't hidden at all, for comparison against a sara run
    under the same disclosure condition.
    """
    policy = POLICIES[policy_name]
    surrogate = surrogate or policy.surrogate
    region = region or policy.region
    sampler = sampler or policy.sampler
    if prior is not None and prior_fixture is not None:
        raise ValueError("pass either prior or prior_fixture, not both")
    if prior_fixture is not None:
        prior = get_prior_fixture(benchmark_name, prior_fixture)
    built = build_sandbox(
        benchmark_name,
        root=root,
        seed=seed,
        reveal=reveal,
        shift=shift,
        context_variant=context_variant,
    )
    sandbox: Path = built["sandbox"]
    state_path = sandbox / "state.json"
    oracle_path = sandbox / "oracle"

    secret = json.loads(built["secret_path"].read_text())
    ob = ObfuscatedBenchmark.from_secret(secret)

    create_args = (
        list(policy.lenz_create_args)
        + ["--surrogate", surrogate, "--budget", str(budget)]
        + list(extra_create_args or [])
    )
    if seed is not None:
        create_args += ["--seed", str(seed)]
    if built["constraints"]:
        create_args += ["--constraints", json.dumps(built["constraints"])]
    n_warm = warmup_n(len(ob.param_names), seed, warmup)
    already = create_and_warmup(
        sandbox, ob.unit_space_json(), create_args, n_warm, budget, objectives=ob.objectives_json()
    )
    _configure_slots(
        state_path,
        region=region,
        sampler=sampler,
        prior=prior,
        decay_beta=decay_beta,
    )

    meta_path = sandbox / "run_meta.json"
    meta = {
        "kind": "lenz",
        "policy": policy_name,
        "budget": budget,
        "seed": seed,
        "warmup": already,
        "surrogate": surrogate,
        "region": region,
        "sampler": sampler,
        "prior": "pibo" if prior else "none",
        "prior_fixture": prior_fixture,
        "prior_belief": prior,
        "prior_decay_beta": decay_beta if prior else None,
        "context_variant": context_variant,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "status": "running",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    try:
        for i in range(already + 1, budget + 1):
            config = lenz(state_path, "suggest")[0]["config"]
            metrics = evaluate(oracle_path, config)
            lenz(state_path, "submit", "--config", json.dumps(config), "--metrics", json.dumps(metrics))
            incumbent_metrics = lenz(state_path, "incumbent")["metrics"]
            best = f"{incumbent_metrics['y']:.4f}" if incumbent_metrics else "none yet (infeasible)"
            print(f"eval {i:3d}/{budget}  y={metrics['y']:.4f}  best-so-far={best}")
    except Exception as e:
        meta.update(ended_at=datetime.now(timezone.utc).isoformat(), status="failed", error=str(e))
        meta_path.write_text(json.dumps(meta, indent=2))
        raise

    meta.update(ended_at=datetime.now(timezone.utc).isoformat(), status="completed")
    meta_path.write_text(json.dumps(meta, indent=2))

    return score_sandbox(built, sandbox)


def main() -> None:
    from .functions import REGISTRY

    p = argparse.ArgumentParser(description="Non-agentic (no sara, no LLM) baselines for the blind benchmark sandbox.")
    p.add_argument(
        "--benchmark",
        required=True,
        help=f"one of {sorted(REGISTRY)}, or 'gp_sample<dim>' (e.g. 'gp_sample6') for a fresh random GP sample path",
    )
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--root", default="./runs/blind")
    p.add_argument("--seed", type=int, default=None, help="pins the renaming/shift transform AND the Sobol warm-start")
    p.add_argument("--warmup", type=int, default=None, help="shared Sobol evaluations before the policy loop (default: d+1 when --seed is set, else 0)")
    p.add_argument("--policy", default="vanilla", choices=sorted(POLICIES), help="see POLICIES in this file")
    p.add_argument("--surrogate", choices=["fixed", "cake"], default=None, help="override the named policy's surrogate slot")
    p.add_argument("--region", choices=["box", "turbo"], default=None, help="override the named policy's region slot")
    p.add_argument("--sampler", choices=["botorch", "llambo"], default=None, help="override the named policy's sampler slot")
    prior_group = p.add_mutually_exclusive_group()
    prior_group.add_argument("--prior", default=None, help="πBO belief JSON")
    prior_group.add_argument("--prior-fixture", default=None, help="named deterministic belief fixture")
    p.add_argument("--decay-beta", type=float, default=10.0, help="πBO decay coefficient")
    p.add_argument("--llm-provider", default=None, help="default plugin LLM for CAKE (no Sara in this script)")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key-env", default=None)
    p.add_argument("--llm-extra-body", default=None)
    p.add_argument("--kernel-llm-provider", default=None, help="CAKE override; else uses --llm-*")
    p.add_argument("--kernel-llm-model", default=None, help="CAKE override; else uses --llm-*")
    p.add_argument("--kernel-llm-base-url", default=None, help="required if --kernel-llm-provider is openai-compatible")
    p.add_argument("--kernel-llm-api-key-env", default=None, help="name of the env var holding the kernel LLM's key -- never the key itself")
    p.add_argument("--kernel-llm-extra-body", default=None, help="JSON object merged into the kernel LLM's request body")
    p.add_argument(
        "--reveal",
        action="store_true",
        help="build a no-blind sandbox instead (real name/bounds/param names) -- a no-op for this script's own "
        "behavior since it never reads context.md, but matches run_noblind_test.py's disclosure condition for "
        "side-by-side comparisons",
    )
    p.add_argument("--shift", action="store_true", help="relocate the optimum even in --reveal mode (default: unmoved)")
    p.add_argument(
        "--context-variant",
        default="domain",
        choices=["domain", "generic", "misleading"],
        help="BoLT context variant; other benchmarks must use domain",
    )
    p.add_argument(
        "--extra-create-arg",
        action="append",
        default=None,
        dest="extra_create_args",
        help="any other raw arg for `lenz create` (repeatable), for one-off tweaks without editing POLICIES",
    )
    args = p.parse_args()

    extra_create_args = list(args.extra_create_args or [])
    if args.llm_provider:
        extra_create_args += ["--llm-provider", args.llm_provider]
    if args.llm_model:
        extra_create_args += ["--llm-model", args.llm_model]
    if args.llm_base_url:
        extra_create_args += ["--llm-base-url", args.llm_base_url]
    if args.llm_api_key_env:
        extra_create_args += ["--llm-api-key-env", args.llm_api_key_env]
    if args.llm_extra_body:
        extra_create_args += ["--llm-extra-body", args.llm_extra_body]
    if args.kernel_llm_provider:
        extra_create_args += ["--kernel-llm-provider", args.kernel_llm_provider]
    if args.kernel_llm_model:
        extra_create_args += ["--kernel-llm-model", args.kernel_llm_model]
    if args.kernel_llm_base_url:
        extra_create_args += ["--kernel-llm-base-url", args.kernel_llm_base_url]
    if args.kernel_llm_api_key_env:
        extra_create_args += ["--kernel-llm-api-key-env", args.kernel_llm_api_key_env]
    if args.kernel_llm_extra_body:
        extra_create_args += ["--kernel-llm-extra-body", args.kernel_llm_extra_body]

    result = run_blind_baseline(
        benchmark_name=args.benchmark,
        budget=args.budget,
        root=Path(args.root),
        policy_name=args.policy,
        seed=args.seed,
        extra_create_args=extra_create_args,
        warmup=args.warmup,
        reveal=args.reveal,
        shift=args.shift,
        context_variant=args.context_variant,
        surrogate=args.surrogate,
        region=args.region,
        sampler=args.sampler,
        prior=json.loads(args.prior) if args.prior else None,
        prior_fixture=args.prior_fixture,
        decay_beta=args.decay_beta,
    )

    print(f"\n=== Blind baseline result ({args.policy}) ===")
    print(f"True benchmark: {result['benchmark']} (revealed post-hoc)")
    print(f"Evaluations observed: {result['n_observed']}")
    print(f"Best true regret: {result['best_regret']:.6f}  (true f_opt = {result['true_f_opt']:.6f})")
    print(f"Sandbox: {result['sandbox']}")


if __name__ == "__main__":
    main()
