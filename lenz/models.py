"""GP surrogate construction. Objective GPs are fit in a positive-is-good
("maximize") convention internally -- `minimize` objectives have their
observed values negated before fitting -- so every acquisition function in
`acquisition.py` can assume maximization. Predictions are converted back to
raw units when reported to the agent.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.warnings import GPInputWarning

from .space import DTYPE, Encoder
from .state import Frame

MIN_POINTS = 2


class ModelError(ValueError):
    pass


def _require_metric(trial, metric: str) -> float:
    if metric not in trial.metrics:
        raise ModelError(f"trial '{trial.trial_id}' is missing metric '{metric}'")
    return float(trial.metrics[metric])


def default_covar_module(d: int) -> Kernel:
    return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))


def fit_gp(X: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor, covar_module: Kernel | None = None) -> SingleTaskGP:
    d = X.shape[-1]
    if covar_module is None:
        covar_module = default_covar_module(d)
    model = SingleTaskGP(
        X,
        y,
        covar_module=covar_module,
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll)
    return model


def bic_score(model: SingleTaskGP, X: torch.Tensor, y: torch.Tensor) -> float:
    """Bayesian information criterion of a fitted GP: lower is a better fit
    penalized for complexity. Used by lenz/cake.py to rank kernel candidates.

    Ported from CAKE's gp.py::fit_gp_model exactly, including its use of
    gpytorch's `ExactMarginalLogLikelihood` output as-is -- which GPyTorch
    normalizes by `num_data` internally (see `ExactMarginalLogLikelihood.forward`).
    This is not the textbook (unnormalized) BIC, but matches the formula CAKE's
    own published results are based on; only relative ranking across the
    population (same `num_data` for every member) matters here.
    """
    model.eval()
    model.likelihood.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with torch.no_grad(), warnings.catch_warnings():
        # cosmetic only: fires because we deliberately evaluate at the training points
        warnings.simplefilter("ignore", GPInputWarning)
        output = model(X)
        log_likelihood = mll(output, y.squeeze(-1)).item()
    num_params = sum(p.numel() for p in model.parameters())
    num_data = X.shape[0]
    return -2 * log_likelihood + num_params * math.log(num_data)


@dataclass
class ModelSet:
    encoder: Encoder
    X: torch.Tensor
    objective_models: dict[str, SingleTaskGP] = field(default_factory=dict)
    constraint_models: dict[str, SingleTaskGP] = field(default_factory=dict)
    objective_sign: dict[str, float] = field(default_factory=dict)
    Y_raw: dict[str, torch.Tensor] = field(default_factory=dict)

    def metric_names(self) -> list[str]:
        return list(self.objective_models) + list(self.constraint_models)


def _covar_for_metric(frame: Frame, d: int, metric: str) -> Kernel | None:
    """Per-metric CAKE kernel when a population exists for that metric.

    Lazy-import cake to avoid a circular import (models -> cake -> acquisition -> models).
    """
    from . import cake

    return cake.covar_module_for_metric(frame, d, metric)


def build_model_set(frame: Frame, encoder: Encoder) -> ModelSet:
    observed = frame.observed_trials()
    if len(observed) < MIN_POINTS:
        raise ModelError("need observed trials")

    X = encoder.stack_features(observed)

    obj_models: dict[str, SingleTaskGP] = {}
    con_models: dict[str, SingleTaskGP] = {}
    signs: dict[str, float] = {}
    y_raw: dict[str, torch.Tensor] = {}

    d = X.shape[-1]

    for o in frame.shelf.objectives:
        y = torch.tensor([[_require_metric(t, o.metric)] for t in observed], dtype=DTYPE)
        sign = -1.0 if o.minimize else 1.0
        signs[o.metric] = sign
        y_raw[o.metric] = y
        obj_models[o.metric] = fit_gp(
            X, y * sign, encoder.domain_bounds, covar_module=_covar_for_metric(frame, d, o.metric)
        )

    for c in frame.shelf.constraints:
        y = torch.tensor([[_require_metric(t, c.metric)] for t in observed], dtype=DTYPE)
        y_raw[c.metric] = y
        con_models[c.metric] = fit_gp(
            X, y, encoder.domain_bounds, covar_module=_covar_for_metric(frame, d, c.metric)
        )

    return ModelSet(
        encoder=encoder,
        X=X,
        objective_models=obj_models,
        constraint_models=con_models,
        objective_sign=signs,
        Y_raw=y_raw,
    )
