"""TuRBO-1 as a region plugin (Eriksson et al., NeurIPS 2019).

Hyperrectangle trust region in GP space. Center tracks the incumbent.
Length doubles after ``success_tolerance`` improvements, halves after
``failure_tolerance`` non-improvements, and restarts at ``length_min``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..space import Encoder
from ..state import Frame, Trial
from .base import SLOT_REGION, LenzPlugin, PluginError

LENGTH_INIT = 0.8
LENGTH_MIN = 0.5 ** 7
LENGTH_MAX = 1.6
SUCCESS_TOLERANCE = 3


def _failure_tolerance(dim: int) -> int:
    return max(4, int(dim))


def default_turbo_state(dim: int = 1) -> dict:
    return {
        "length": LENGTH_INIT,
        "length_init": LENGTH_INIT,
        "length_min": LENGTH_MIN,
        "length_max": LENGTH_MAX,
        "success_tolerance": SUCCESS_TOLERANCE,
        "failure_tolerance": _failure_tolerance(dim),
        "success_count": 0,
        "failure_count": 0,
        "restarts": 0,
        "center": None,
        "best_value": None,
    }


class TurboPlugin(LenzPlugin):
    name = "turbo"
    slot = SLOT_REGION

    def default_state(self) -> dict:
        return default_turbo_state()

    def add_parser(self, sub, state_parent) -> None:
        p = sub.add_parser("turbo", parents=[state_parent])
        inner = p.add_subparsers(dest="turbo_cmd", required=True)
        inner.add_parser("status")
        inner.add_parser("init")
        ov = inner.add_parser("override")
        ov.add_argument("--center", default=None, help="JSON config to use as TR center")
        ov.add_argument("--length", type=float, default=None)

    def commands(self):
        return {"turbo": self._dispatch}

    def on_observe(self, frame: Frame, trial: Trial) -> None:
        if frame.shelf.region != "turbo":
            return
        encoder = Encoder(frame.space)
        self.ensure(frame, encoder)
        self._update(frame, encoder, trial)

    def active_bounds(self, frame: Frame, encoder: Encoder) -> torch.Tensor | None:
        if frame.shelf.region != "turbo":
            return None
        blob = self.ensure(frame, encoder)
        center_cfg = blob.get("center")
        if not center_cfg:
            return None
        center = encoder.encode(center_cfg)
        length = float(blob["length"])
        lo = encoder.domain_bounds[0]
        hi = encoder.domain_bounds[1]
        width = (hi - lo).clamp_min(1e-12)
        half = 0.5 * length * width
        tr_lo = torch.clamp(center - half, min=lo)
        tr_hi = torch.clamp(center + half, max=hi)
        # Keep a non-empty interval on collapsed dims.
        too_small = tr_hi - tr_lo < 1e-12 * width
        tr_hi = torch.where(too_small, torch.clamp(tr_lo + 1e-6 * width, max=hi), tr_hi)
        return torch.stack([tr_lo, tr_hi])

    def summary(self, frame: Frame) -> dict:
        blob = dict(self.blob(frame))
        return {
            "length": blob.get("length"),
            "success_count": blob.get("success_count"),
            "failure_count": blob.get("failure_count"),
            "restarts": blob.get("restarts"),
            "center": blob.get("center"),
            "best_value": blob.get("best_value"),
        }

    def prompt_path(self) -> Path | None:
        path = Path(__file__).parent / "turbo.md"
        return path if path.exists() else None

    def ensure(self, frame: Frame, encoder: Encoder) -> dict:
        blob = self.blob(frame)
        dim = encoder.d
        blob.setdefault("failure_tolerance", _failure_tolerance(dim))
        if blob.get("center") is None:
            center, value = _incumbent_center(frame, encoder)
            if center is not None:
                blob["center"] = center
                blob["best_value"] = value
        return blob

    def _dispatch(self, frame: Frame, args):
        encoder = Encoder(frame.space)
        cmd = args.turbo_cmd
        if cmd == "init":
            frame.plugins["turbo"] = default_turbo_state(encoder.d)
            frame.shelf.region = "turbo"
            self.ensure(frame, encoder)
            frame.log_event("turbo", action="init")
            return frame, self.summary(frame)
        if cmd == "status":
            if frame.shelf.region == "turbo":
                self.ensure(frame, encoder)
            return None, {"region": frame.shelf.region, **self.summary(frame)}
        if cmd == "override":
            blob = self.ensure(frame, encoder)
            if args.center:
                import json

                blob["center"] = json.loads(args.center)
            if args.length is not None:
                blob["length"] = float(args.length)
            frame.shelf.region = "turbo"
            frame.log_event("turbo", action="override")
            return frame, self.summary(frame)
        raise PluginError(f"unknown turbo subcommand '{cmd}'")

    def _update(self, frame: Frame, encoder: Encoder, trial: Trial) -> None:
        blob = self.blob(frame)
        improved, value = _trial_improves(frame, trial, blob.get("best_value"))
        if improved:
            blob["success_count"] = int(blob.get("success_count") or 0) + 1
            blob["failure_count"] = 0
            blob["center"] = dict(trial.config)
            blob["best_value"] = value
            if blob["success_count"] >= int(blob["success_tolerance"]):
                blob["length"] = min(float(blob["length_max"]), 2.0 * float(blob["length"]))
                blob["success_count"] = 0
        else:
            blob["failure_count"] = int(blob.get("failure_count") or 0) + 1
            blob["success_count"] = 0
            if blob["failure_count"] >= int(blob["failure_tolerance"]):
                blob["length"] = 0.5 * float(blob["length"])
                blob["failure_count"] = 0
        if float(blob["length"]) < float(blob["length_min"]):
            blob["length"] = float(blob["length_init"])
            blob["success_count"] = 0
            blob["failure_count"] = 0
            blob["restarts"] = int(blob.get("restarts") or 0) + 1
            center, value = _incumbent_center(frame, encoder)
            if center is not None:
                blob["center"] = center
                blob["best_value"] = value
        frame.log_event(
            "turbo",
            action="update",
            length=blob["length"],
            success_count=blob["success_count"],
            failure_count=blob["failure_count"],
            restarts=blob["restarts"],
        )


def _signed_value(frame: Frame, metrics: dict) -> float | None:
    if frame.shelf.is_moo or not frame.shelf.objectives:
        return None
    obj = frame.shelf.objectives[0]
    if metrics is None or obj.metric not in metrics:
        return None
    sign = -1.0 if obj.minimize else 1.0
    return float(metrics[obj.metric]) * sign


def _incumbent_center(frame: Frame, encoder: Encoder) -> tuple[dict | None, float | None]:
    from .. import optimize as opt
    from .. import acquisition as acq

    if frame.shelf.is_moo:
        pareto = opt.get_pareto(frame)
        if not pareto:
            return None, None
        return dict(pareto[0].config), None
    trial = opt.get_incumbent(frame, encoder)
    if trial is None:
        observed = [
            t
            for t in frame.observed_trials()
            if acq.config_feasible(frame.shelf.constraints, t.metrics)
        ]
        if not observed:
            return None, None
        trial = observed[-1]
    return dict(trial.config), _signed_value(frame, trial.metrics)


def _trial_improves(frame: Frame, trial: Trial, best_value: float | None) -> tuple[bool, float | None]:
    from .. import acquisition as acq

    if trial.metrics is None:
        return False, best_value
    if not acq.config_feasible(frame.shelf.constraints, trial.metrics):
        return False, best_value
    value = _signed_value(frame, trial.metrics)
    if value is None:
        return False, best_value
    if best_value is None:
        return True, value
    return value > float(best_value) + 1e-12, max(value, float(best_value))
