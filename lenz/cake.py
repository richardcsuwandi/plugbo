"""CAKE (Context-Aware Kernel Evolution) as a lenz surrogate-design module.

Implements the NeurIPS 2025 algorithm: population management, BIC fitness,
LLM-guided crossover and mutation, and BAKER ranking. LLM calls go through
`llm.factory`, and BAKER's acquisition half follows whatever acqf lenz has
configured (logEI, NEHVI/EHVI, constrained EI, feasibility) instead of
hardcoded analytic EI.

Each objective metric and each constraint metric gets its own kernel population.
BAKER ranks weighted kernel combinations: for each combo it fits the chosen
kernels, optimizes the configured acquisition function, and scores
prod(softmax(-BIC)) * acqf(x*). Combo count is capped; with one target this
reduces to classic single-objective BAKER.

lenz owns the outer BO loop; CAKE only ever answers two questions -- "should
the population evolve now" and "given the population, which kernel/query
pair does BAKER rank highest" -- both exposed through `commands.py`, never
by CAKE picking or committing an evaluation itself.
"""

from __future__ import annotations

import itertools
import math
import random
import threading
from collections.abc import Callable

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from llm.base import LLMClient, Message
from .llm_config import client_from_spec, resolved_plugin_llm, spec_complete, spec_from_args

from .acquisition import (
    KNOWN_ACQFS,
    AcqfError,
    ProbabilityOfFeasibility,
    build_moo_acqf,
    build_single_objective_acqf,
    has_feasible_incumbent,
)
from .kernels import DEFAULT_POPULATION, KernelParseError, parse_kernel_expression
from .models import ModelSet, bic_score, fit_gp
from .space import DTYPE, Encoder
from .state import Frame, Objective

OPERATORS = ["+", "*"]

_CAKE_DEFAULTS = {
    "kernel_llm": {},
    "kernel_population_size": 6,
    "kernel_init_after": 6,
    "kernel_evolve_every": 4,
    "kernel_freeze_fraction": 0.5,
    "kernel_num_crossover": 1,
    "kernel_mutation_prob": 0.7,
    "kernel_populations": {},
    "kernel_evolution_states": {},
}


def default_state() -> dict:
    return {
        "kernel_llm": {},
        "kernel_population_size": 6,
        "kernel_init_after": 6,
        "kernel_evolve_every": 4,
        "kernel_freeze_fraction": 0.5,
        "kernel_num_crossover": 1,
        "kernel_mutation_prob": 0.7,
        "kernel_populations": {},
        "kernel_evolution_states": {},
    }


def state(frame: Frame) -> dict:
    """CAKE's namespaced blob under ``frame.plugins['cake']``."""
    blob = frame.plugins.setdefault("cake", default_state())
    for key, value in _CAKE_DEFAULTS.items():
        if key not in blob:
            blob[key] = dict(value) if isinstance(value, dict) else value
    return blob


def apply_kernel_args(frame: Frame, args) -> None:
    """Writes --kernel-* / --budget flags into the CAKE blob (and shelf.budget)."""
    blob = state(frame)
    if getattr(args, "budget", None) is not None:
        frame.shelf.budget = args.budget
    spec = spec_from_args(args, "kernel_llm")
    if spec_complete(spec):
        blob["kernel_llm"] = spec
    for key in (
        "kernel_population_size",
        "kernel_init_after",
        "kernel_evolve_every",
        "kernel_freeze_fraction",
        "kernel_num_crossover",
        "kernel_mutation_prob",
    ):
        value = getattr(args, key, None)
        if value is not None:
            blob[key] = value


NUM_RESTARTS = 8
RAW_SAMPLES = 128
MAX_KERNEL_EXPR_LENGTH = 10  # CAKE bloat-control check (paper: len(kernel) < 10)
BAKER_MAX_COMBOS = 32  # cap on kernel tuples evaluated per suggest/score call

# Kernel evolution is best-effort background maintenance triggered synchronously inside
# `observe`/`submit` -- a slow or hung LLM call must not block the whole call indefinitely.
# One evolution round makes up to 2 sequential calls (crossover + mutation), so this bounds
# a single `observe` call to roughly 2x this value in the worst case.
KERNEL_LLM_TIMEOUT_SECONDS = 90.0


class LLMCallTimeout(TimeoutError):
    pass


def _call_with_hard_timeout(fn, timeout_seconds: float):
    """Enforces a real wall-clock deadline on `fn()`, regardless of what the
    HTTP client's own `timeout=` does. Observed in practice: a streaming
    response's read-timeout is measured between chunks, not for the whole
    request, so a slow trickle can evade a client-level timeout entirely
    while still taking many minutes. Runs `fn` in a daemon thread so an
    abandoned call can never block this process from exiting.
    """
    box: dict = {}
    done = threading.Event()

    def run():
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001 -- re-raised on the calling thread below
            box["error"] = e
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(timeout_seconds):
        raise LLMCallTimeout(f"LLM call exceeded {timeout_seconds}s hard timeout")
    if "error" in box:
        raise box["error"]
    return box["value"]


class CakeNotReadyError(RuntimeError):
    """surrogate == 'cake' but no population exists yet (schedule not due)."""


class LLMResponseParseError(ValueError):
    pass


def cake_targets(frame: Frame) -> list[str]:
    """Objective metrics plus constraint metrics, each with its own CAKE population."""
    seen: set[str] = set()
    out: list[str] = []
    for o in frame.shelf.objectives:
        if o.metric not in seen:
            seen.add(o.metric)
            out.append(o.metric)
    for c in frame.shelf.constraints:
        if c.metric not in seen:
            seen.add(c.metric)
            out.append(c.metric)
    return out


def can_use_baker(frame: Frame) -> bool:
    """True when CAKE surrogate is active and BAKER applies (not pure Sobol)."""
    return frame.shelf.surrogate == "cake" and frame.shelf.acqf != "sobol"


def _feasibility_phase(frame: Frame) -> bool:
    return bool(frame.shelf.constraints) and not has_feasible_incumbent(frame)


def _baker_targets(frame: Frame) -> list[str]:
    """Metrics whose kernel populations BAKER enumerates for this suggest/score."""
    if _feasibility_phase(frame):
        return [c.metric for c in frame.shelf.constraints]
    return cake_targets(frame)


def _format_kernel_label(kernels: dict[str, str]) -> str:
    if len(kernels) == 1:
        return next(iter(kernels.values()))
    return "|".join(f"{m}:{expr}" for m, expr in sorted(kernels.items()))


def _model_set_for_kernels(
    frame: Frame, encoder: Encoder, X: torch.Tensor, observed: list, kernels: dict[str, str]
) -> ModelSet:
    obj_models: dict[str, SingleTaskGP] = {}
    con_models: dict[str, SingleTaskGP] = {}
    signs: dict[str, float] = {}
    y_raw: dict[str, torch.Tensor] = {}

    for o in frame.shelf.objectives:
        if o.metric not in kernels:
            continue
        y, yr, sign = _metric_tensors(frame, observed, o.metric)
        obj_models[o.metric] = _fit_kernel_model(X, y, encoder, kernels[o.metric])
        signs[o.metric] = sign
        y_raw[o.metric] = yr

    for c in frame.shelf.constraints:
        if c.metric not in kernels:
            continue
        y, yr, _ = _metric_tensors(frame, observed, c.metric)
        con_models[c.metric] = _fit_kernel_model(X, y, encoder, kernels[c.metric])
        y_raw[c.metric] = yr

    return ModelSet(
        encoder=encoder,
        X=X,
        objective_models=obj_models,
        constraint_models=con_models,
        objective_sign=signs,
        Y_raw=y_raw,
    )


def _prepared_populations(
    frame: Frame, encoder: Encoder, X: torch.Tensor, observed: list, targets: list[str]
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Returns [(metric, [(expression, softmax_weight), ...]), ...] sorted by BIC."""
    out: list[tuple[str, list[tuple[str, float]]]] = []
    for target in targets:
        population = state(frame)["kernel_populations"].get(target) or []
        if not population:
            raise CakeNotReadyError
        y, _, _ = _metric_tensors(frame, observed, target)
        _refresh_fitness(population, encoder, X, y)
        viable = [m for m in population if _finite_bic(m)]
        if not viable:
            raise CakeNotReadyError
        weights = _population_weights(viable)
        ranked = sorted(zip(viable, weights), key=lambda pair: pair[0]["bic"])
        out.append((target, [(m["expression"], w) for m, w in ranked]))
    return out


def _kernel_combos(
    prepared: list[tuple[str, list[tuple[str, float]]]], max_combos: int = BAKER_MAX_COMBOS
) -> list[tuple[dict[str, str], float]]:
    """Kernel tuples to evaluate. One target -> classic BAKER over its population."""
    if not prepared:
        raise CakeNotReadyError

    n = len(prepared)
    k = max(1, int(max_combos ** (1.0 / n)))
    trimmed = [choices[: min(k, len(choices))] for _, choices in prepared]

    combos: list[tuple[dict[str, str], float]] = []
    for prod_item in itertools.product(*trimmed):
        metrics = [m for m, _ in prepared]
        kernels = {metrics[i]: prod_item[i][0] for i in range(n)}
        weight = math.prod(prod_item[i][1] for i in range(n))
        combos.append((kernels, weight))

    if len(combos) > max_combos:
        combos.sort(key=lambda c: c[1], reverse=True)
        combos = combos[:max_combos]
    return combos


def _build_baker_acqf(
    model_set: ModelSet,
    frame: Frame,
    acqf_name: str,
    X_pending: torch.Tensor | None,
):
    if acqf_name == "probability_of_feasibility":
        return ProbabilityOfFeasibility(model_set, frame), acqf_name
    if frame.shelf.is_moo:
        return build_moo_acqf(model_set, frame, acqf_name, X_pending), acqf_name
    return build_single_objective_acqf(model_set, frame, acqf_name, frame.shelf.acqf_params, X_pending), acqf_name


def _baker_acqf_name(frame: Frame) -> str:
    if _feasibility_phase(frame):
        return "probability_of_feasibility"
    return frame.shelf.acqf


def _population(frame: Frame, target: str) -> list[dict]:
    return state(frame)["kernel_populations"].setdefault(target, [])


def _evolution_state(frame: Frame, target: str) -> dict:
    states = state(frame)["kernel_evolution_states"]
    if target not in states:
        states[target] = {"generation": 0, "last_evolved_at_n_observed": 0, "frozen": False}
    return states[target]


def _objective_for_metric(frame: Frame, target: str) -> Objective | None:
    for o in frame.shelf.objectives:
        if o.metric == target:
            return o
    return None


def _metric_tensors(
    frame: Frame, observed: list, target: str
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (y_for_gp_fit, y_raw, sign). Objectives are sign-adjusted to maximize."""
    obj = _objective_for_metric(frame, target)
    y_raw = torch.tensor([[float(t.metrics[target])] for t in observed], dtype=DTYPE)
    if obj is None:
        return y_raw, y_raw, 1.0
    sign = -1.0 if obj.minimize else 1.0
    return y_raw * sign, y_raw, sign


# -- prompts (response format: "Kernel: ... / Analysis: ...") ----------------

_KERNEL_SYSTEM_PROMPT = """You are an expert in machine learning, specializing in Gaussian processes. Here are the observations we have collected so far:
{observations}

Please analyze these observations to identify patterns in the data that can be captured by a kernel function.
You can use any of the following base kernels: {base_kernels}, and combine these kernels using the following operators: {operators}.
Your goal is to construct a kernel expression that best explains the observed data.
The kernel will be evaluated using a fitness score normalized between [0, 1], where higher values indicate better fit to the data.

Respond in exactly this format, with no other text:
Kernel: <kernel expression, e.g. "M5 + PER">
Analysis: <one or two sentences of reasoning>
"""

_CROSSOVER_PROMPT = """You are given two parent kernels and their fitness scores:
{parent_kernel1} ({fitness1}), {parent_kernel2} ({fitness2})

Please propose a new kernel that has a potentially higher fitness score.
You may combine the parent kernels using any of the operators from: {operators}.
Briefly explain your reasoning behind the proposed kernel.
"""

_MUTATION_PROMPT = """You are given a kernel and its fitness score:
{kernel} ({fitness})

Please propose a new kernel that has a potentially higher fitness score.
You may replace a base kernel in the current expression with another base kernel from the set: {base_kernels}.
Briefly explain your reasoning behind the proposed kernel.
"""


def _parse_llm_response(response: str) -> tuple[str, str]:
    if "Kernel:" not in response or "Analysis:" not in response:
        raise LLMResponseParseError(f"response missing 'Kernel:'/'Analysis:' markers: {response[:200]!r}")
    kernel_start = response.index("Kernel:") + len("Kernel:")
    kernel_end = response.find("\n", kernel_start)
    if kernel_end == -1:
        kernel_end = len(response)
    kernel = response[kernel_start:kernel_end].strip()
    analysis_start = response.index("Analysis:") + len("Analysis:")
    analysis = response[analysis_start:].strip()
    return kernel, analysis


def _system_prompt(X: torch.Tensor, y: torch.Tensor) -> str:
    observations = "\n".join(f"x = {xi}, y = {yi[0]}" for xi, yi in zip(X.tolist(), y.tolist()))
    return _KERNEL_SYSTEM_PROMPT.format(
        observations=observations, base_kernels=DEFAULT_POPULATION, operators=OPERATORS
    )


# -- fitting helpers ---------------------------------------------------------


def _finite_bic(member: dict) -> bool:
    b = member.get("bic")
    return b is not None and math.isfinite(b)


def _fit_kernel_model(X: torch.Tensor, y: torch.Tensor, encoder: Encoder, expression: str) -> SingleTaskGP:
    kernel = parse_kernel_expression(expression, X.shape[-1])
    return fit_gp(X, y, encoder.domain_bounds, covar_module=kernel)


def _refresh_fitness(
    population: list[dict], encoder: Encoder, X: torch.Tensor, y: torch.Tensor
) -> None:
    """Refits every population member against the current data and updates
    its BIC in place. `y` is already sign-adjusted to the maximize convention.
    """
    for member in population:
        try:
            model = _fit_kernel_model(X, y, encoder, member["expression"])
            member["bic"] = bic_score(model, X, y)
        except Exception:
            member["bic"] = float("inf")


def _population_weights(population: list[dict]) -> list[float]:
    bics = torch.tensor([m["bic"] for m in population], dtype=torch.double)
    if bics.numel() > 1 and bics.std() > 0:
        bics = (bics - bics.mean()) / bics.std()
    return torch.softmax(-bics, dim=0).tolist()


def _upsert_member(population: list[dict], expression: str, bic: float, generation: int) -> None:
    for m in population:
        if m["expression"] == expression:
            m["bic"] = bic
            m["generation"] = generation
            return
    population.append({"expression": expression, "bic": bic, "generation": generation})


# -- evolution (crossover + mutation + survivor selection) -----------------


def _crossover_step(
    population: list[dict],
    frame: Frame,
    encoder: Encoder,
    client: LLMClient,
    X: torch.Tensor,
    y: torch.Tensor,
    sys_prompt: str,
    generation: int,
) -> None:
    for _ in range(state(frame)["kernel_num_crossover"]):
        if len(population) < 2:
            return
        viable = [m for m in population if _finite_bic(m)]
        if len(viable) < 2:
            return
        weights = _population_weights(viable)
        i1, i2 = np.random.choice(len(viable), size=2, replace=False, p=weights)
        parent1, parent2 = viable[int(i1)], viable[int(i2)]

        try:
            resp = _call_with_hard_timeout(
                lambda: client.chat(
                    [
                        Message(
                            role="user",
                            content=_CROSSOVER_PROMPT.format(
                                parent_kernel1=parent1["expression"],
                                fitness1=parent1["bic"],
                                parent_kernel2=parent2["expression"],
                                fitness2=parent2["bic"],
                                operators=OPERATORS,
                            ),
                        )
                    ],
                    tools=[],
                    system=sys_prompt,
                ),
                KERNEL_LLM_TIMEOUT_SECONDS,
            )
            expression, _ = _parse_llm_response(resp.content)
        except Exception:
            expression = f"{parent1['expression']} {random.choice(OPERATORS)} {parent2['expression']}"

        if len(expression) >= MAX_KERNEL_EXPR_LENGTH:
            continue
        try:
            model = _fit_kernel_model(X, y, encoder, expression)
            bic = bic_score(model, X, y)
        except Exception:
            continue
        _upsert_member(population, expression, bic, generation)


def _mutation_step(
    population: list[dict],
    frame: Frame,
    encoder: Encoder,
    client: LLMClient,
    X: torch.Tensor,
    y: torch.Tensor,
    sys_prompt: str,
    generation: int,
) -> None:
    if not population or random.random() >= state(frame)["kernel_mutation_prob"]:
        return
    viable = [m for m in population if _finite_bic(m)]
    if not viable:
        return
    fittest = min(viable, key=lambda m: m["bic"])
    try:
        resp = _call_with_hard_timeout(
            lambda: client.chat(
                [
                    Message(
                        role="user",
                        content=_MUTATION_PROMPT.format(
                            kernel=fittest["expression"], fitness=fittest["bic"], base_kernels=DEFAULT_POPULATION
                        ),
                    )
                ],
                tools=[],
                system=sys_prompt,
            ),
            KERNEL_LLM_TIMEOUT_SECONDS,
        )
        expression, _ = _parse_llm_response(resp.content)
        model = _fit_kernel_model(X, y, encoder, expression)
        bic = bic_score(model, X, y)
    except Exception:
        return
    _upsert_member(population, expression, bic, generation)


def _select_survivors(population: list[dict], frame: Frame) -> None:
    population.sort(key=lambda m: m["bic"] if m["bic"] is not None else float("inf"))
    del population[state(frame)["kernel_population_size"] :]


def evolve_generation(frame: Frame, encoder: Encoder, client: LLMClient, target: str) -> None:
    """One generation for `target`: fitness -> crossover -> mutation -> survivor selection.
    Lazily seeds the population with the six base kernels on first call
    (CAKE's own `run()` treats "initialize" and "evolve" as the same step).
    """
    observed = frame.observed_trials()
    X = encoder.stack_features(observed)
    y, _, _ = _metric_tensors(frame, observed, target)
    population = _population(frame, target)
    evo = _evolution_state(frame, target)

    if not population:
        state(frame)["kernel_populations"][target] = [
            {"expression": name, "bic": None, "generation": 0} for name in DEFAULT_POPULATION
        ]
        population = _population(frame, target)

    generation = evo.get("generation", 0) + 1
    sys_prompt = _system_prompt(X, y)

    _refresh_fitness(population, encoder, X, y)
    _crossover_step(population, frame, encoder, client, X, y, sys_prompt, generation)
    _mutation_step(population, frame, encoder, client, X, y, sys_prompt, generation)
    _refresh_fitness(population, encoder, X, y)
    _select_survivors(population, frame)

    evo["generation"] = generation
    evo["last_evolved_at_n_observed"] = len(observed)
    frame.log_event(
        "evolve-kernels",
        target=target,
        generation=generation,
        population=[m["expression"] for m in population],
        best=get_best_kernel(frame, target),
    )


def get_best_kernel(frame: Frame, target: str | None = None) -> str | None | dict[str, str | None]:
    if target is None:
        return {t: get_best_kernel(frame, t) for t in cake_targets(frame)}
    population = state(frame)["kernel_populations"].get(target) or []
    viable = [m for m in population if _finite_bic(m)]
    if not viable:
        return None
    return min(viable, key=lambda m: m["bic"])["expression"]


def covar_module_for_metric(frame: Frame, d: int, metric: str):
    """Best-BIC kernel expression for `metric`, or None if unavailable."""
    if frame.shelf.surrogate != "cake":
        return None
    population = state(frame)["kernel_populations"].get(metric) or []
    viable = [m for m in population if _finite_bic(m)]
    if not viable:
        return None
    best = min(viable, key=lambda m: m["bic"])
    try:
        return parse_kernel_expression(best["expression"], d)
    except KernelParseError:
        return None


# -- scheduling ---------------------------------------------------------


def should_evolve(frame: Frame, target: str) -> bool:
    """Mechanical, decoupled from Sara's reasoning cadence: lenz checks this
    itself at the end of every successful `observe`/`submit --metrics`.
    """
    if frame.shelf.surrogate != "cake":
        return False
    evo = _evolution_state(frame, target)
    if evo.get("frozen"):
        return False

    n_observed = len(frame.observed_trials())
    budget = frame.shelf.budget
    blob = state(frame)
    if budget and n_observed >= budget * blob["kernel_freeze_fraction"]:
        evo["frozen"] = True  # freeze from here on; population is kept, just no more LLM calls
        return False

    population = blob["kernel_populations"].get(target) or []
    if not population:
        return n_observed >= blob["kernel_init_after"]

    last = evo.get("last_evolved_at_n_observed", 0)
    return n_observed - last >= blob["kernel_evolve_every"]


def maybe_evolve(frame: Frame, encoder: Encoder, force: bool = False) -> bool:
    """Checks (unless `force`) whether evolution is due and, if so, runs one
    generation per target metric. Best-effort: LLM/parsing failures are logged,
    never raised. Returns whether any generation actually ran.
    """
    if frame.shelf.surrogate != "cake":
        return False
    if len(frame.observed_trials()) < 2:
        return False

    llm_cfg = resolved_plugin_llm(frame, state(frame).get("kernel_llm"))
    if not spec_complete(llm_cfg):
        frame.log_event("evolve-kernels", status="skipped", reason="kernel_llm not configured")
        return False

    try:
        client = client_from_spec(llm_cfg, timeout=KERNEL_LLM_TIMEOUT_SECONDS)
    except Exception as e:
        frame.log_event("evolve-kernels", status="failed", error=str(e))
        return False

    ran_any = False
    for target in cake_targets(frame):
        if not force and not should_evolve(frame, target):
            continue
        try:
            evolve_generation(frame, encoder, client, target)
            ran_any = True
        except Exception as e:
            frame.log_event("evolve-kernels", status="failed", target=target, error=str(e))
    return ran_any


# -- BAKER: weighted kernel x acquisition ranking ---------------------------


def _eval_acqf(acqf, x: torch.Tensor) -> float:
    with torch.no_grad():
        val = acqf(x.unsqueeze(0).unsqueeze(0))
    return float(val.squeeze().item())


def baker_suggest(
    frame: Frame,
    encoder: Encoder,
    q: int,
    X_pending: torch.Tensor | None,
    *,
    bounds: torch.Tensor | None = None,
    wrap_acqf: Callable | None = None,
) -> list[dict]:
    """Rank kernel combinations by prod(softmax(-BIC)) * acqf(x*), then return
    the top-`q` (kernel combo, query) pairs. Uses NEHVI/EHVI for MOO, constrained
    logEI when constraints are present, and probability-of-feasibility before any
    feasible incumbent exists. One target reduces to classic single-objective BAKER.
    """
    observed = frame.observed_trials()
    X = encoder.stack_features(observed)
    bounds = bounds if bounds is not None else encoder.encode_bounds(frame.shelf.bounds)
    targets = _baker_targets(frame)
    prepared = _prepared_populations(frame, encoder, X, observed, targets)
    combos = _kernel_combos(prepared)
    acqf_name = _baker_acqf_name(frame)

    candidates = []
    for kernels, weight in combos:
        try:
            model_set = _model_set_for_kernels(frame, encoder, X, observed, kernels)
            acqf, _ = _build_baker_acqf(model_set, frame, acqf_name, X_pending)
            if wrap_acqf is not None and acqf_name != "probability_of_feasibility":
                acqf = wrap_acqf(acqf)
            x_star, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES
            )
            acq_val = _eval_acqf(acqf, x_star.squeeze(0))
        except Exception:
            continue
        candidates.append(
            {
                "kernels": kernels,
                "x": x_star.squeeze(0),
                "acq_val": acq_val,
                "score": weight * acq_val,
            }
        )

    if not candidates:
        raise CakeNotReadyError

    candidates.sort(key=lambda c: c["score"], reverse=True)
    out = []
    for c in candidates[:q]:
        label = _format_kernel_label(c["kernels"])
        out.append(
            {
                "config": encoder.decode(c["x"]),
                "x_gp": c["x"].detach().tolist(),
                "acquisition_values": {acqf_name: c["acq_val"], "baker_score": c["score"]},
                "trial_id": None,
                "acqf": acqf_name,
                "kernel": label,
                "kernels": c["kernels"],
            }
        )
    return out


def baker_score(
    frame: Frame,
    encoder: Encoder,
    configs: list[dict],
    acqf_names: list[str],
    *,
    wrap_acqf: Callable | None = None,
) -> list[dict]:
    """Weighted-ensemble BAKER score at fixed configs: for each acqf, sum over
    kernel combos of prod(softmax(-BIC)) * acqf(combo, x).
    """
    observed = frame.observed_trials()
    X = encoder.stack_features(observed)
    targets = _baker_targets(frame)
    prepared = _prepared_populations(frame, encoder, X, observed, targets)
    combos = _kernel_combos(prepared)
    Xc = torch.stack([encoder.encode(c) for c in configs])

    results: list[dict] = [dict() for _ in configs]
    for name in acqf_names:
        if name not in KNOWN_ACQFS and name != "probability_of_feasibility":
            raise AcqfError(f"unknown acqf '{name}'")
        if name == "sobol":
            for r in results:
                r[name] = 0.0
            continue

        totals = [0.0] * len(configs)
        any_ok = False
        for kernels, weight in combos:
            try:
                model_set = _model_set_for_kernels(frame, encoder, X, observed, kernels)
                acqf, _ = _build_baker_acqf(model_set, frame, name, None)
                if wrap_acqf is not None and name != "probability_of_feasibility":
                    acqf = wrap_acqf(acqf)
            except Exception:
                continue
            any_ok = True
            for i in range(len(configs)):
                totals[i] += weight * _eval_acqf(acqf, Xc[i])

        if not any_ok:
            raise CakeNotReadyError
        for i, r in enumerate(results):
            r[name] = totals[i]

    return results
