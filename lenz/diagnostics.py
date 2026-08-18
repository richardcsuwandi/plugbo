"""Surrogate diagnostics: leave-one-out CV R^2, noise, lengthscales, sensitivity."""

from __future__ import annotations

import random

import torch

from .models import ModelSet, fit_gp
from .space import Encoder, RangeDim

MAX_CV_POINTS = 25


def leave_one_out_r2(X: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor) -> float:
    n = X.shape[0]
    if n < 3:
        return float("nan")
    idxs = list(range(n))
    if n > MAX_CV_POINTS:
        idxs = random.sample(idxs, MAX_CV_POINTS)
    preds, actuals = [], []
    for i in idxs:
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        try:
            m = fit_gp(X[mask], y[mask], bounds)
        except Exception:
            continue
        with torch.no_grad():
            pred = m.posterior(X[i : i + 1]).mean.squeeze().item()
        preds.append(pred)
        actuals.append(y[i].item())
    if len(preds) < 2:
        return float("nan")
    preds_t = torch.tensor(preds)
    actuals_t = torch.tensor(actuals)
    ss_res = ((actuals_t - preds_t) ** 2).sum()
    ss_tot = ((actuals_t - actuals_t.mean()) ** 2).sum()
    if ss_tot <= 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def unstandardized_noise(model) -> float:
    noise = model.likelihood.noise.mean().item()
    ot = getattr(model, "outcome_transform", None)
    if ot is not None and hasattr(ot, "stdvs"):
        std = ot.stdvs.flatten()[0].item()
        noise = noise * (std**2)
    return float(noise)


def lengthscales_by_dim(model, encoder: Encoder) -> dict[str, float]:
    ls = model.covar_module.base_kernel.lengthscale.detach().flatten()
    out = {}
    for name in encoder.space.names:
        sl = encoder._col_slices[name]
        out[name] = float(ls[sl].mean().item())
    return out


def sensitivity_by_dim(model, encoder: Encoder, X: torch.Tensor) -> dict[str, float]:
    """Mean posterior-mean gradient per dimension, scaled by that dimension's
    domain width so results are comparable across differently-scaled dims.
    Signed for range/ordered dims (positive = increasing it helps); reported
    as magnitude for unordered categoricals, which have no single direction.
    """
    Xg = X.clone().requires_grad_(True)
    mean = model.posterior(Xg).mean.sum()
    (grad,) = torch.autograd.grad(mean, Xg)
    grad = grad.mean(dim=0)
    out = {}
    for name, dim in encoder.space.dims.items():
        sl = encoder._col_slices[name]
        width = encoder.domain_bounds[1, sl] - encoder.domain_bounds[0, sl]
        contrib = grad[sl] * width
        signed = isinstance(dim, RangeDim) or getattr(dim, "ordered", False)
        out[name] = float(contrib.sum().item()) if signed else float(contrib.abs().sum().item())
    return out


def compute_diagnostics(model_set: ModelSet, encoder: Encoder) -> dict:
    result: dict = {"n_observed": model_set.X.shape[0], "objectives": {}, "constraints": {}}
    for metric, model in model_set.objective_models.items():
        y = model_set.Y_raw[metric] * model_set.objective_sign[metric]
        result["objectives"][metric] = {
            "cv_r2": leave_one_out_r2(model_set.X, y, encoder.domain_bounds),
            "noise": unstandardized_noise(model),
            "lengthscales": lengthscales_by_dim(model, encoder),
            "sensitivity": sensitivity_by_dim(model, encoder, model_set.X),
        }
    for metric, model in model_set.constraint_models.items():
        y = model_set.Y_raw[metric]
        result["constraints"][metric] = {
            "cv_r2": leave_one_out_r2(model_set.X, y, encoder.domain_bounds),
            "noise": unstandardized_noise(model),
            "lengthscales": lengthscales_by_dim(model, encoder),
            "sensitivity": sensitivity_by_dim(model, encoder, model_set.X),
        }
    return result
