"""suggest / score / predict / incumbent / pareto logic: ties together the
search space encoder, GP models, and acquisition functions.
"""

from __future__ import annotations

import torch
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples

from . import acquisition as acq
from . import cake as cake_module
from .models import MIN_POINTS, build_model_set
from .plugins.registry import occupant
from .space import DTYPE, Encoder
from .state import Frame, Trial

NUM_RESTARTS = 8
RAW_SAMPLES = 128


class OptimizeError(ValueError):
    pass


def needs_warmup(frame: Frame, encoder: Encoder) -> bool:
    threshold = max(MIN_POINTS, encoder.d + 1)
    return len(frame.observed_trials()) < threshold


def active_bounds(frame: Frame, encoder: Encoder) -> torch.Tensor:
    region = occupant(frame, "region")
    if region is not None:
        bounds = region.active_bounds(frame, encoder)
        if bounds is not None:
            return bounds
    return encoder.encode_bounds(frame.shelf.bounds)


def _sobol_x(
    encoder: Encoder, bounds: torch.Tensor, q: int, seed: int | None = None, skip: int = 0
) -> tuple[torch.Tensor, int]:
    """Continuous GP-space Sobol draws. Projection onto mixed types happens in `decode`."""
    d = int(bounds.shape[-1])
    if seed is None:
        samples = draw_sobol_samples(bounds=bounds, n=1, q=q).squeeze(0)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        return samples, skip + q
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seed))
    if skip:
        engine.fast_forward(skip)
    unit = engine.draw(q).to(dtype=bounds.dtype)
    lo, hi = bounds[0], bounds[1]
    samples = lo + (hi - lo) * unit
    return samples, skip + q


def sobol_candidates(
    encoder: Encoder, bounds: torch.Tensor, q: int, seed: int | None = None, skip: int = 0
) -> tuple[list[dict], int]:
    """Returns (configs, new_skip). When `seed` is set, draws are a scrambled
    Sobol sequence that continues from `skip`, so sequential q=1 calls match
    a single q=n call. Unseeded draws stay independently scrambled (old behavior).
    """
    samples, new_skip = _sobol_x(encoder, bounds, q, seed=seed, skip=skip)
    return [encoder.decode(samples[i]) for i in range(q)], new_skip


def _pending_X(frame: Frame, encoder: Encoder) -> torch.Tensor | None:
    pending = frame.in_flight_trials()
    if not pending:
        return None
    return encoder.stack_features(pending)


def _pack_suggestions(
    encoder: Encoder,
    X: torch.Tensor,
    *,
    acqf_name: str,
    acquisition_values: list[dict] | None = None,
) -> list[dict]:
    out = []
    for i in range(X.shape[0]):
        xi = X[i]
        payload = {
            "config": encoder.decode(xi),
            "x_gp": xi.detach().tolist(),
            "acquisition_values": {} if acquisition_values is None else acquisition_values[i],
            "trial_id": None,
            "acqf": acqf_name,
        }
        out.append(payload)
    return out


def _sobol_suggestions(frame: Frame, encoder: Encoder, bounds: torch.Tensor, q: int) -> list[dict]:
    samples, drawn = _sobol_x(encoder, bounds, q, seed=frame.shelf.seed, skip=frame.shelf.sobol_drawn)
    frame.shelf.sobol_drawn = drawn
    return _pack_suggestions(encoder, samples, acqf_name="sobol")


def _cake_population_all_unfittable(frame: Frame) -> bool:
    """True when CAKE has a population but every member has a non-finite BIC.

    In that case we must not silently swap in the default Matérn: that would
    make the cake surrogate a no-op. Suggest falls back to Sobol for the step.
    """
    if frame.shelf.surrogate != "cake":
        return False
    targets = cake_module.cake_targets(frame)
    if not targets:
        return False
    any_pop = False
    for target in targets:
        pop = cake_module.state(frame)["kernel_populations"].get(target) or []
        if not pop:
            continue
        any_pop = True
        if cake_module.get_best_kernel(frame, target) is not None:
            return False
    return any_pop


def _wrap_acqf(acqf, frame: Frame, encoder: Encoder):
    prior = occupant(frame, "prior")
    if prior is None:
        return acqf
    return prior.wrap_acqf(acqf, frame, encoder)


def _eval_acqf(acqf, x: torch.Tensor) -> float:
    with torch.no_grad():
        val = acqf(x.unsqueeze(0).unsqueeze(0))
    return float(val.squeeze().item())


def get_incumbent(frame: Frame, encoder: Encoder, in_bounds: bool = False) -> Trial | None:
    if frame.shelf.is_moo:
        raise OptimizeError("incumbent is single-objective only; use 'pareto' for multi-objective studies")
    objective = frame.shelf.objectives[0]
    sign = -1.0 if objective.minimize else 1.0
    bounds = active_bounds(frame, encoder) if in_bounds else None
    best_trial, best_val = None, -float("inf")
    for t in frame.observed_trials():
        if not acq.config_feasible(frame.shelf.constraints, t.metrics):
            continue
        if bounds is not None:
            x = encoder.encode(t.config)
            if bool(((x < bounds[0] - 1e-9) | (x > bounds[1] + 1e-9)).any()):
                continue
        val = float(t.metrics[objective.metric]) * sign
        if val > best_val:
            best_val, best_trial = val, t
    return best_trial


def get_pareto(frame: Frame) -> list[Trial]:
    if not frame.shelf.is_moo:
        raise OptimizeError("pareto is multi-objective only; use 'incumbent' for single-objective studies")
    feasible = [t for t in frame.observed_trials() if acq.config_feasible(frame.shelf.constraints, t.metrics)]
    if not feasible:
        return []
    metrics = [o.metric for o in frame.shelf.objectives]
    signs = [-1.0 if o.minimize else 1.0 for o in frame.shelf.objectives]
    Y = torch.tensor(
        [[float(t.metrics[m]) * s for m, s in zip(metrics, signs)] for t in feasible], dtype=DTYPE
    )
    mask = is_non_dominated(Y)
    return [t for t, keep in zip(feasible, mask.tolist()) if keep]


def _commit_suggestions(frame: Frame, suggestions: list[dict]) -> list[dict]:
    frame.clear_pending_x_gp()
    for item in suggestions:
        frame.remember_suggestion(item["config"], item.get("x_gp"))
    return suggestions


def suggest(
    frame: Frame,
    encoder: Encoder,
    q: int = 1,
    bounds_override: dict | None = None,
    around: dict | float | None = None,
    radius: float = 0.1,
) -> list[dict]:
    if bounds_override is not None:
        merged = {**frame.shelf.bounds, **bounds_override}
        bounds = encoder.encode_bounds(merged)
    elif around is not None:
        inc = get_incumbent(frame, encoder) if not frame.shelf.is_moo else None
        if inc is None and frame.shelf.is_moo:
            pareto = get_pareto(frame)
            inc = pareto[0] if pareto else None
        if inc is None:
            raise OptimizeError("need observed trials")
        per_dim = around if isinstance(around, dict) else None
        r = around if isinstance(around, (int, float)) else radius
        bounds = encoder.radius_bounds(inc.config, radius=r, per_dim=per_dim)
    else:
        bounds = active_bounds(frame, encoder)

    plain_call = bounds_override is None and around is None
    if plain_call:
        sampler = occupant(frame, "sampler")
        if sampler is not None:
            proposed = sampler.propose(frame, encoder, q, bounds)
            if proposed is not None:
                return _commit_suggestions(frame, proposed)

    if plain_call and cake_module.can_use_baker(frame) and not needs_warmup(frame, encoder):
        try:
            return _commit_suggestions(
                frame,
                cake_module.baker_suggest(
                    frame,
                    encoder,
                    q,
                    _pending_X(frame, encoder),
                    bounds=bounds,
                    wrap_acqf=lambda acqf: _wrap_acqf(acqf, frame, encoder),
                ),
            )
        except cake_module.CakeNotReadyError:
            pass  # populations not ready -- fall back to best kernel per metric below

    if frame.shelf.acqf == "sobol" or needs_warmup(frame, encoder) or _cake_population_all_unfittable(frame):
        return _commit_suggestions(frame, _sobol_suggestions(frame, encoder, bounds, q))

    model_set = build_model_set(frame, encoder)
    X_pending = _pending_X(frame, encoder)

    if frame.shelf.constraints and not acq.has_feasible_incumbent(frame):
        acqf = acq.ProbabilityOfFeasibility(model_set, frame)
        name = "probability_of_feasibility"
        candidates = []
        for _ in range(q):
            X, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=1, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES
            )
            candidates.append(X.squeeze(0))
        X = torch.stack(candidates)
    else:
        name = frame.shelf.acqf
        if frame.shelf.is_moo:
            acqf = acq.build_moo_acqf(model_set, frame, name, X_pending)
        else:
            acqf = acq.build_single_objective_acqf(model_set, frame, name, frame.shelf.acqf_params, X_pending)
        acqf = _wrap_acqf(acqf, frame, encoder)
        X, _ = optimize_acqf(
            acq_function=acqf, bounds=bounds, q=q, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES
        )

    vals = [_eval_acqf(acqf, X[i]) for i in range(q)]
    return _commit_suggestions(
        frame,
        _pack_suggestions(encoder, X, acqf_name=name, acquisition_values=[{name: v} for v in vals]),
    )


def score(frame: Frame, encoder: Encoder, configs: list[dict], acqf_names: list[str]) -> list[dict]:
    if frame.shelf.surrogate == "cake" and cake_module.can_use_baker(frame):
        try:
            return cake_module.baker_score(
                frame,
                encoder,
                configs,
                acqf_names,
                wrap_acqf=lambda acqf: _wrap_acqf(acqf, frame, encoder),
            )
        except cake_module.CakeNotReadyError:
            pass  # populations not ready -- fall back to best kernel per metric below

    model_set = build_model_set(frame, encoder)
    X_pending = _pending_X(frame, encoder)
    results = [dict() for _ in configs]
    for name in acqf_names:
        if name not in acq.KNOWN_ACQFS:
            raise acq.AcqfError(f"unknown acqf '{name}'")
        if name == "sobol":
            for r in results:
                r[name] = 0.0
            continue
        if frame.shelf.is_moo:
            acqf = acq.build_moo_acqf(model_set, frame, name, X_pending)
        else:
            acqf = acq.build_single_objective_acqf(model_set, frame, name, frame.shelf.acqf_params, X_pending)
        acqf = _wrap_acqf(acqf, frame, encoder)
        for cfg, r in zip(configs, results):
            r[name] = _eval_acqf(acqf, encoder.encode(cfg))
    return results


def predict(frame: Frame, encoder: Encoder, configs: list[dict]) -> list[dict]:
    model_set = build_model_set(frame, encoder)
    X = torch.stack([encoder.encode(c) for c in configs])
    out = [{"mean": {}, "variance": {}} for _ in configs]
    for metric, model in model_set.objective_models.items():
        sign = model_set.objective_sign[metric]
        with torch.no_grad():
            posterior = model.posterior(X)
            mean = posterior.mean.squeeze(-1) * sign
            var = posterior.variance.squeeze(-1)
        for i in range(len(configs)):
            out[i]["mean"][metric] = float(mean[i].item())
            out[i]["variance"][metric] = float(var[i].item())
    for metric, model in model_set.constraint_models.items():
        with torch.no_grad():
            posterior = model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            var = posterior.variance.squeeze(-1)
        for i in range(len(configs)):
            out[i]["mean"][metric] = float(mean[i].item())
            out[i]["variance"][metric] = float(var[i].item())
    if frame.shelf.constraints:
        with torch.no_grad():
            probs = acq.feasibility_prob(model_set, frame, X)
        for i in range(len(configs)):
            out[i]["prob_feasible"] = float(probs[i].item())
    return out
