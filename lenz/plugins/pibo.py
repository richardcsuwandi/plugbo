"""πBO: prior-weighted acquisition (Hvarfner et al., 2022).

α_π(x) = α(x) * π(x)^{β / (t+1)}. Sara compiles a factorized belief with
``lenz set-belief``; this plugin occupies the prior slot and wraps the
configured BoTorch acquisition.
"""

from __future__ import annotations

from pathlib import Path

import torch
from botorch.acquisition import AcquisitionFunction
from torch.distributions import Beta, LogNormal, Normal

from ..space import ChoiceDim, Encoder, RangeDim
from ..state import Frame
from .base import SLOT_PRIOR, LenzPlugin, PluginError

PRIOR_FLOOR = 1e-12
DEFAULT_DECAY_BETA = 10.0


class PiboPlugin(LenzPlugin):
    name = "pibo"
    slot = SLOT_PRIOR

    def default_state(self) -> dict:
        return {"belief": {}, "decay_beta": DEFAULT_DECAY_BETA, "t0": 0, "prior_floor": PRIOR_FLOOR}

    def add_parser(self, sub, state_parent) -> None:
        p = sub.add_parser("set-belief", parents=[state_parent])
        p.add_argument("--prior", required=True, help="JSON object mapping parameter -> distribution spec")
        p.add_argument("--decay-beta", type=float, default=None)
        p.add_argument("--clear", action="store_true")

    def commands(self):
        return {"set-belief": self._set_belief}

    def wrap_acqf(self, acqf, frame: Frame, encoder: Encoder):
        if frame.shelf.prior != "pibo":
            return acqf
        blob = self.blob(frame)
        if not blob.get("belief"):
            return acqf
        return PriorWeightedAcquisition(
            acqf,
            frame,
            encoder,
            blob["belief"],
            float(blob.get("decay_beta") or DEFAULT_DECAY_BETA),
            int(blob.get("t0") or 0),
            float(blob.get("prior_floor") or PRIOR_FLOOR),
        )

    def summary(self, frame: Frame) -> dict:
        blob = self.blob(frame)
        return {
            "belief": blob.get("belief") or {},
            "decay_beta": blob.get("decay_beta"),
            "t0": blob.get("t0"),
        }

    def prompt_path(self) -> Path | None:
        path = Path(__file__).parent / "pibo.md"
        return path if path.exists() else None

    def _set_belief(self, frame: Frame, args):
        import json

        blob = self.blob(frame)
        if args.clear:
            blob["belief"] = {}
            frame.shelf.prior = "none"
            frame.log_event("set-belief", cleared=True)
            return frame, self.summary(frame)
        belief = json.loads(args.prior)
        _validate_belief(frame, belief)
        blob["belief"] = belief
        if args.decay_beta is not None:
            blob["decay_beta"] = float(args.decay_beta)
        blob["t0"] = len(frame.observed_trials())
        frame.shelf.prior = "pibo"
        frame.log_event("set-belief", n_params=len(belief))
        return frame, self.summary(frame)


def _validate_belief(frame: Frame, belief: dict) -> None:
    for name, spec in belief.items():
        if name not in frame.space.dims:
            raise PluginError(f"belief references unknown dimension '{name}'")
        if not isinstance(spec, dict) or "dist" not in spec:
            raise PluginError(f"belief for '{name}' must be an object with 'dist'")
        dist = spec["dist"]
        dim = frame.space.dims[name]
        if dist in ("normal", "lognormal", "beta", "uniform"):
            if not isinstance(dim, RangeDim):
                raise PluginError(f"belief for '{name}': '{dist}' requires a range dimension")
        elif dist == "categorical":
            if not isinstance(dim, ChoiceDim):
                raise PluginError(f"belief for '{name}': categorical requires a choice dimension")
        else:
            raise PluginError(f"unknown dist '{dist}' for '{name}'")


def prior_density(frame: Frame, config: dict, belief: dict, floor: float = PRIOR_FLOOR) -> float:
    """Factorized π(x) = ∏_i π_i(x_i), with a floor so the product stays positive."""
    density = 1.0
    for name, spec in belief.items():
        if name not in config:
            continue
        density *= max(_one_pdf(frame, name, config[name], spec), floor)
    return max(density, floor)


def _one_pdf(frame: Frame, name: str, value, spec: dict) -> float:
    dist = spec["dist"]
    dim = frame.space.dims[name]
    if dist == "uniform":
        return 1.0
    if dist == "normal":
        mu = float(spec["mu"])
        sigma = float(spec.get("sigma", 1.0))
        return float(Normal(mu, max(sigma, 1e-9)).log_prob(torch.tensor(float(value))).exp())
    if dist == "lognormal":
        mu = float(spec["mu"])
        sigma = float(spec.get("sigma", 1.0))
        x = max(float(value), 1e-12)
        return float(LogNormal(mu, max(sigma, 1e-9)).log_prob(torch.tensor(x)).exp())
    if dist == "beta":
        a = float(spec.get("a", spec.get("alpha", 2.0)))
        b = float(spec.get("b", spec.get("beta", 2.0)))
        if not isinstance(dim, RangeDim):
            return 1.0
        width = dim.upper - dim.lower
        u = (float(value) - dim.lower) / width if width else 0.5
        u = min(max(u, 1e-6), 1.0 - 1e-6)
        return float(Beta(a, b).log_prob(torch.tensor(u)).exp()) / max(width, 1e-12)
    if dist == "categorical":
        probs = spec.get("probs") or {}
        return float(probs.get(value, PRIOR_FLOOR))
    return 1.0


class PriorWeightedAcquisition(AcquisitionFunction):
    def __init__(
        self,
        base: AcquisitionFunction,
        frame: Frame,
        encoder: Encoder,
        belief: dict,
        decay_beta: float,
        t0: int,
        prior_floor: float,
    ):
        super().__init__(model=base.model)
        self.base = base
        self.frame = frame
        self.encoder = encoder
        self.belief = belief
        self.decay_beta = decay_beta
        self.t0 = t0
        self.prior_floor = prior_floor
        self._rescale = frame.shelf.acqf in ("ucb", "pi")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        acq_values = self.base(X)
        t = max(0, len(self.frame.observed_trials()) - self.t0)
        exponent = self.decay_beta / (t + 1)
        prior = torch.ones(X.shape[:-1], dtype=acq_values.dtype, device=acq_values.device)
        for b in range(X.shape[0]):
            for q in range(X.shape[1]):
                cfg = self.encoder.decode(X[b, q].detach())
                prior[b, q] = prior_density(self.frame, cfg, self.belief, self.prior_floor)
        if prior.ndim > acq_values.ndim:
            prior = prior.prod(dim=-1)
        log_prior = prior.clamp_min(self.prior_floor).log()
        # logEI / log-HVI are already logs. Product α * π^β in the original
        # space is a sum in log space. Multiplying a negative logEI by a large
        # prior would invert the ranking.
        if self.frame.shelf.acqf in {"noisy_logei", "logei", "nehvi", "ehvi"}:
            return acq_values + exponent * log_prior
        if self._rescale:
            eta = 0.0
            if self.frame.shelf.objectives:
                metric = self.frame.shelf.objectives[0].metric
                vals = [
                    float(t.metrics[metric])
                    for t in self.frame.observed_trials()
                    if t.metrics and metric in t.metrics
                ]
                if vals:
                    eta = abs(max(vals) if not self.frame.shelf.objectives[0].minimize else -min(vals))
            acq_values = (acq_values + eta).clamp_min(0.0)
        return acq_values * prior.clamp_min(self.prior_floor).pow(exponent)
