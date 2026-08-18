"""Acquisition function construction. Objective models are already fit in a
positive-is-good convention (see models.py), so every acquisition function
here simply maximizes.
"""

from __future__ import annotations

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch.distributions import Normal

from .models import ModelSet
from .state import Constraint, Frame

SAMPLES = 256


class AcqfError(ValueError):
    pass


KNOWN_ACQFS = {"noisy_logei", "logei", "pi", "ucb", "sobol", "nehvi", "ehvi"}


def _sampler() -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(sample_shape=torch.Size([SAMPLES]))


def _constraint_callables(constraints: list[Constraint], offset: int) -> list:
    """Build one callable per (metric, bound) pair on posterior samples, feasible
    when the callable's return value is <= 0, indexing into output columns
    starting at `offset` (the constraint models follow the objective(s)).
    """
    fns = []
    for i, c in enumerate(constraints):
        col = offset + i
        if c.upper is not None:
            fns.append(lambda samples, col=col, ub=c.upper: samples[..., col] - ub)
        if c.lower is not None:
            fns.append(lambda samples, col=col, lb=c.lower: lb - samples[..., col])
    return fns


def feasibility_prob(model_set: ModelSet, frame: Frame, X: torch.Tensor) -> torch.Tensor:
    """Analytic, independence-assumed joint probability that all constraints
    hold at each row of X (shape (n, d)). Returns shape (n,).
    """
    normal = Normal(0.0, 1.0)
    n = X.shape[0]
    log_prob = torch.zeros(n, dtype=X.dtype)
    for c in frame.shelf.constraints:
        model = model_set.constraint_models[c.metric]
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        p = torch.ones(n, dtype=X.dtype)
        if c.upper is not None:
            p = p * normal.cdf((c.upper - mean) / std)
        if c.lower is not None:
            p = p * (1 - normal.cdf((c.lower - mean) / std))
        log_prob = log_prob + torch.log(p.clamp_min(1e-12))
    return log_prob.exp()


class ProbabilityOfFeasibility(AcquisitionFunction):
    """Maximizes joint probability of feasibility; used before any feasible
    incumbent has been observed for a constrained problem. Only supports q=1.
    """

    def __init__(self, model_set: ModelSet, frame: Frame):
        super().__init__(model=next(iter(model_set.constraint_models.values())))
        self.model_set = model_set
        self.frame = frame

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-2] != 1:
            raise AcqfError("ProbabilityOfFeasibility only supports q=1")
        return feasibility_prob(self.model_set, self.frame, X.squeeze(-2))


def has_feasible_incumbent(frame: Frame) -> bool:
    if not frame.shelf.constraints:
        return True
    for t in frame.observed_trials():
        if config_feasible(frame.shelf.constraints, t.metrics):
            return True
    return False


def config_feasible(constraints: list[Constraint], metrics: dict) -> bool:
    for c in constraints:
        v = float(metrics[c.metric])
        if c.upper is not None and v > c.upper:
            return False
        if c.lower is not None and v < c.lower:
            return False
    return True


def build_single_objective_acqf(
    model_set: ModelSet,
    frame: Frame,
    name: str,
    params: dict,
    X_pending: torch.Tensor | None,
) -> AcquisitionFunction:
    obj_metric = frame.shelf.objectives[0].metric
    obj_model = model_set.objective_models[obj_metric]
    constraints_spec = frame.shelf.constraints

    y_sign = model_set.Y_raw[obj_metric] * model_set.objective_sign[obj_metric]
    if constraints_spec:
        con_metrics = list(model_set.constraint_models)
        model = ModelListGP(obj_model, *[model_set.constraint_models[m] for m in con_metrics])
        objective = GenericMCObjective(lambda samples, X=None: samples[..., 0])
        constraints = _constraint_callables(constraints_spec, offset=1)
    else:
        model = obj_model
        objective = None
        constraints = None

    best_f = float(y_sign.max().item())
    if constraints_spec:
        feas_vals = [
            y_sign[i].item()
            for i, t in enumerate(frame.observed_trials())
            if config_feasible(constraints_spec, t.metrics)
        ]
        if feas_vals:
            best_f = max(feas_vals)

    sampler = _sampler()

    if name == "noisy_logei":
        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=model_set.X,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            prune_baseline=True,
        )
    if name == "logei":
        return qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
    if name == "pi":
        return qProbabilityOfImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
    if name == "ucb":
        beta = float(params.get("beta", 2.0))
        return qUpperConfidenceBound(
            model=model, beta=beta, sampler=sampler, objective=objective, X_pending=X_pending
        )
    raise AcqfError(f"unknown acqf '{name}'")


def ref_point_for_moo(model_set: ModelSet, frame: Frame, margin: float = 0.1) -> torch.Tensor:
    metrics = [o.metric for o in frame.shelf.objectives]
    cols = []
    for m in metrics:
        y = model_set.Y_raw[m] * model_set.objective_sign[m]
        lo, hi = float(y.min()), float(y.max())
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        cols.append(lo - margin * span)
    return torch.tensor(cols, dtype=model_set.X.dtype)


def build_moo_acqf(
    model_set: ModelSet,
    frame: Frame,
    name: str,
    X_pending: torch.Tensor | None,
) -> AcquisitionFunction:
    metrics = [o.metric for o in frame.shelf.objectives]
    obj_models = [model_set.objective_models[m] for m in metrics]
    n_obj = len(metrics)
    constraints_spec = frame.shelf.constraints

    if constraints_spec:
        con_metrics = list(model_set.constraint_models)
        model = ModelListGP(*obj_models, *[model_set.constraint_models[m] for m in con_metrics])
        objective = IdentityMCMultiOutputObjective(outcomes=list(range(n_obj)))
        constraints = _constraint_callables(constraints_spec, offset=n_obj)
    else:
        model = ModelListGP(*obj_models)
        objective = None
        constraints = None

    ref_point = ref_point_for_moo(model_set, frame)
    Y = torch.stack([model_set.Y_raw[m].squeeze(-1) * model_set.objective_sign[m] for m in metrics], dim=-1)
    sampler = _sampler()

    if name == "nehvi":
        return qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=model_set.X,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            prune_baseline=True,
        )
    if name == "ehvi":
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y)
        return qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
        )
    raise AcqfError(f"unknown multi-objective acqf '{name}' (use 'nehvi' or 'ehvi')")
