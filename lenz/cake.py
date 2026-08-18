"""CAKE (Context-Aware Kernel Evolution) as a lenz surrogate-design module.

Implements the NeurIPS 2025 algorithm: population management, BIC fitness,
LLM-guided crossover and mutation, and BAKER ranking. LLM calls go through
`llm.factory`, and BAKER's acquisition half follows whatever acqf lenz has
configured instead of hardcoded analytic EI.

lenz owns the outer BO loop; CAKE only ever answers two questions -- "should
the population evolve now" and "given the population, which kernel/query
pair does BAKER rank highest" -- both exposed through `commands.py`, never
by CAKE picking or committing an evaluation itself.
"""

from __future__ import annotations

import os
import random
import threading

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from llm.base import LLMClient, Message
from llm.factory import get_client

from .acquisition import KNOWN_ACQFS, AcqfError, build_single_objective_acqf
from .kernels import DEFAULT_POPULATION, KernelParseError, parse_kernel_expression
from .models import ModelSet, bic_score, fit_gp
from .space import DTYPE, Encoder
from .state import Frame

OPERATORS = ["+", "*"]
NUM_RESTARTS = 8
RAW_SAMPLES = 128
MAX_KERNEL_EXPR_LENGTH = 10  # CAKE bloat-control check (paper: len(kernel) < 10)

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


def _fit_kernel_model(X: torch.Tensor, y: torch.Tensor, encoder: Encoder, expression: str) -> SingleTaskGP:
    kernel = parse_kernel_expression(expression, X.shape[-1])
    return fit_gp(X, y, encoder.domain_bounds, covar_module=kernel)


def _refresh_fitness(frame: Frame, encoder: Encoder, X: torch.Tensor, y: torch.Tensor) -> None:
    """Refits every population member against the current data and updates
    its BIC in place. `y` is already sign-adjusted to the maximize convention.
    """
    for member in frame.shelf.kernel_population:
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
    frame: Frame, encoder: Encoder, client: LLMClient, X: torch.Tensor, y: torch.Tensor, sys_prompt: str, generation: int
) -> None:
    population = frame.shelf.kernel_population
    for _ in range(frame.shelf.kernel_num_crossover):
        if len(population) < 2:
            return
        weights = _population_weights(population)
        i1, i2 = np.random.choice(len(population), size=2, replace=False, p=weights)
        parent1, parent2 = population[int(i1)], population[int(i2)]

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
    frame: Frame, encoder: Encoder, client: LLMClient, X: torch.Tensor, y: torch.Tensor, sys_prompt: str, generation: int
) -> None:
    population = frame.shelf.kernel_population
    if not population or random.random() >= frame.shelf.kernel_mutation_prob:
        return
    fittest = min(population, key=lambda m: m["bic"] if m["bic"] is not None else float("inf"))
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


def _select_survivors(frame: Frame) -> None:
    population = frame.shelf.kernel_population
    population.sort(key=lambda m: m["bic"] if m["bic"] is not None else float("inf"))
    del population[frame.shelf.kernel_population_size :]


def evolve_generation(frame: Frame, encoder: Encoder, client: LLMClient) -> None:
    """One generation: fitness -> crossover -> mutation -> survivor selection.
    Lazily seeds the population with the six base kernels on first call
    (CAKE's own `run()` treats "initialize" and "evolve" as the same step).
    """
    observed = frame.observed_trials()
    obj = frame.shelf.objectives[0]
    X = torch.stack([encoder.encode(t.config) for t in observed])
    sign = -1.0 if obj.minimize else 1.0
    y = torch.tensor([[float(t.metrics[obj.metric])] for t in observed], dtype=DTYPE) * sign

    if not frame.shelf.kernel_population:
        frame.shelf.kernel_population = [
            {"expression": name, "bic": None, "generation": 0} for name in DEFAULT_POPULATION
        ]

    generation = frame.shelf.kernel_evolution_state.get("generation", 0) + 1
    sys_prompt = _system_prompt(X, y)

    _refresh_fitness(frame, encoder, X, y)
    _crossover_step(frame, encoder, client, X, y, sys_prompt, generation)
    _mutation_step(frame, encoder, client, X, y, sys_prompt, generation)
    _refresh_fitness(frame, encoder, X, y)
    _select_survivors(frame)

    frame.shelf.kernel_evolution_state["generation"] = generation
    frame.shelf.kernel_evolution_state["last_evolved_at_n_observed"] = len(observed)
    frame.log_event(
        "evolve-kernels",
        generation=generation,
        population=[m["expression"] for m in frame.shelf.kernel_population],
        best=get_best_kernel(frame),
    )


def get_best_kernel(frame: Frame) -> str | None:
    population = frame.shelf.kernel_population
    if not population:
        return None
    return min(population, key=lambda m: m["bic"] if m["bic"] is not None else float("inf"))["expression"]


# -- scheduling ---------------------------------------------------------


def should_evolve(frame: Frame) -> bool:
    """Mechanical, decoupled from Sara's reasoning cadence: lenz checks this
    itself at the end of every successful `observe`/`submit --metrics`.
    """
    if frame.shelf.surrogate != "cake":
        return False
    state = frame.shelf.kernel_evolution_state
    if state.get("frozen"):
        return False

    n_observed = len(frame.observed_trials())
    budget = frame.shelf.budget
    if budget and n_observed >= budget * frame.shelf.kernel_freeze_fraction:
        state["frozen"] = True  # freeze from here on; population is kept, just no more LLM calls
        return False

    if not frame.shelf.kernel_population:
        return n_observed >= frame.shelf.kernel_init_after

    last = state.get("last_evolved_at_n_observed", 0)
    return n_observed - last >= frame.shelf.kernel_evolve_every


def maybe_evolve(frame: Frame, encoder: Encoder, force: bool = False) -> bool:
    """Checks (unless `force`) whether evolution is due and, if so, runs one
    generation. Best-effort: LLM/parsing failures are logged, never raised --
    a transient hiccup here must not fail the `observe`/`evolve-kernels` call
    that triggered it. Returns whether a generation actually ran.
    """
    if not force and not should_evolve(frame):
        return False
    if frame.shelf.surrogate != "cake":
        return False
    if len(frame.observed_trials()) < 2:
        return False

    llm_cfg = frame.shelf.kernel_llm
    provider, model = llm_cfg.get("provider"), llm_cfg.get("model")
    if not provider or not model:
        frame.log_event("evolve-kernels", status="skipped", reason="kernel_llm not configured")
        return False

    api_key = os.environ.get(llm_cfg["api_key_env"]) if llm_cfg.get("api_key_env") else None
    try:
        client = get_client(
            provider,
            model,
            base_url=llm_cfg.get("base_url"),
            api_key=api_key,
            timeout=KERNEL_LLM_TIMEOUT_SECONDS,
            extra_body=llm_cfg.get("extra_body"),
        )
        evolve_generation(frame, encoder, client)
        return True
    except Exception as e:
        frame.log_event("evolve-kernels", status="failed", error=str(e))
        return False


# -- BAKER: weighted kernel x acquisition ranking ---------------------------


def _eval_acqf(acqf, x: torch.Tensor) -> float:
    with torch.no_grad():
        val = acqf(x.unsqueeze(0).unsqueeze(0))
    return float(val.squeeze().item())


def baker_suggest(frame: Frame, encoder: Encoder, q: int, X_pending: torch.Tensor | None) -> list[dict]:
    """`weight_k = softmax(-BIC_k)`, `score_k = weight_k * acqf_k(x*_k)` under
    lenz's currently configured acqf; returns the top-`q` kernel/query pairs.
    """
    if not frame.shelf.kernel_population:
        raise CakeNotReadyError

    observed = frame.observed_trials()
    obj = frame.shelf.objectives[0]
    X = torch.stack([encoder.encode(t.config) for t in observed])
    sign = -1.0 if obj.minimize else 1.0
    y_raw = torch.tensor([[float(t.metrics[obj.metric])] for t in observed], dtype=DTYPE)
    y = y_raw * sign

    _refresh_fitness(frame, encoder, X, y)
    population = frame.shelf.kernel_population
    weights = _population_weights(population)
    bounds = encoder.encode_bounds(frame.shelf.bounds)
    acqf_name = frame.shelf.acqf

    candidates = []
    for member, weight in zip(population, weights):
        try:
            model = _fit_kernel_model(X, y, encoder, member["expression"])
            model_set = ModelSet(
                encoder=encoder,
                X=X,
                objective_models={obj.metric: model},
                constraint_models={},
                objective_sign={obj.metric: sign},
                Y_raw={obj.metric: y_raw},
            )
            acqf = build_single_objective_acqf(model_set, frame, acqf_name, frame.shelf.acqf_params, X_pending)
            x_star, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES
            )
            acq_val = _eval_acqf(acqf, x_star.squeeze(0))
        except Exception:
            continue
        candidates.append(
            {"expression": member["expression"], "x": x_star.squeeze(0), "acq_val": acq_val, "score": weight * acq_val}
        )

    if not candidates:
        raise CakeNotReadyError

    candidates.sort(key=lambda c: c["score"], reverse=True)
    out = []
    for c in candidates[:q]:
        out.append(
            {
                "config": encoder.decode(c["x"]),
                "acquisition_values": {acqf_name: c["acq_val"], "baker_score": c["score"]},
                "trial_id": None,
                "acqf": acqf_name,
                "kernel": c["expression"],
            }
        )
    return out


def baker_score(frame: Frame, encoder: Encoder, configs: list[dict], acqf_names: list[str]) -> list[dict]:
    """Weighted-ensemble version of `score`: for each acqf, sums each
    population member's acquisition value at the given candidates, weighted
    by `softmax(-BIC)` -- keeps `score` apples-to-apples with `baker_suggest`.
    """
    if not frame.shelf.kernel_population:
        raise CakeNotReadyError

    observed = frame.observed_trials()
    obj = frame.shelf.objectives[0]
    X = torch.stack([encoder.encode(t.config) for t in observed])
    sign = -1.0 if obj.minimize else 1.0
    y_raw = torch.tensor([[float(t.metrics[obj.metric])] for t in observed], dtype=DTYPE)
    y = y_raw * sign

    _refresh_fitness(frame, encoder, X, y)
    population = frame.shelf.kernel_population
    weights = _population_weights(population)
    Xc = torch.stack([encoder.encode(c) for c in configs])

    results: list[dict] = [dict() for _ in configs]
    for name in acqf_names:
        if name not in KNOWN_ACQFS:
            raise AcqfError(f"unknown acqf '{name}'")
        if name == "sobol":
            for r in results:
                r[name] = 0.0
            continue

        totals = [0.0] * len(configs)
        any_ok = False
        for member, weight in zip(population, weights):
            try:
                model = _fit_kernel_model(X, y, encoder, member["expression"])
                model_set = ModelSet(
                    encoder=encoder,
                    X=X,
                    objective_models={obj.metric: model},
                    constraint_models={},
                    objective_sign={obj.metric: sign},
                    Y_raw={obj.metric: y_raw},
                )
                acqf = build_single_objective_acqf(model_set, frame, name, frame.shelf.acqf_params, None)
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
