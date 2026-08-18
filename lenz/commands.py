"""Implementation of each lenz CLI command. Every function takes a parsed
argparse.Namespace (plus a loaded Frame where applicable) and returns
`(frame_or_None, result_dict)`. `cli.py` handles the JSON envelope, state
load/save, and error formatting.
"""

from __future__ import annotations

import json

from . import optimize as opt
from .acquisition import KNOWN_ACQFS, AcqfError
from .diagnostics import compute_diagnostics
from .llm_config import (
    apply_sidecar_plugin_overrides,
    core_spec_from_sidecar,
    load_sidecar,
    spec_complete,
    spec_from_args,
)
from .models import build_model_set
from .plugins.base import PluginError
from .plugins.registry import (
    all_plugins,
    enabled_plugins,
    occupant,
    plugin_by_name,
    plugins_for_slot,
    surrogate_names,
)
from .space import Encoder, SearchSpace
from .state import Constraint, Frame, Objective, Shelf, StateError

MOO_ONLY_ACQFS = {"nehvi", "ehvi"}
SINGLE_ONLY_ACQFS = {"noisy_logei", "logei", "pi", "ucb"}


class SurrogateError(ValueError):
    pass


def _check_surrogate_compat(surrogate: str, is_moo: bool, has_constraints: bool) -> None:
    del is_moo, has_constraints
    if surrogate not in surrogate_names():
        raise SurrogateError(f"unknown surrogate '{surrogate}' (expected {surrogate_names()})")


class NoMatchingSubmission(StateError):
    def __init__(self, outstanding: list[dict]):
        self.outstanding = outstanding
        super().__init__("config matches no submitted point")


def _check_acqf_compat(acqf: str, is_moo: bool) -> None:
    if acqf not in KNOWN_ACQFS:
        raise AcqfError(f"unknown acqf '{acqf}'")
    if is_moo and acqf in SINGLE_ONLY_ACQFS:
        raise AcqfError(f"multi-objective study cannot use '{acqf}'; use 'nehvi', 'ehvi', or 'sobol'")
    if not is_moo and acqf in MOO_ONLY_ACQFS:
        raise AcqfError(f"single-objective study cannot use '{acqf}'; use 'noisy_logei', 'logei', 'pi', 'ucb', or 'sobol'")


def _objectives_json(frame: Frame) -> list[dict]:
    return [{"metric": o.metric, "minimize": o.minimize} for o in frame.shelf.objectives]


def _constraints_json(frame: Frame) -> list[dict]:
    return [{"metric": c.metric, "lower": c.lower, "upper": c.upper} for c in frame.shelf.constraints]


def _parse_objectives(raw: dict) -> list[Objective]:
    if not raw:
        raise StateError("objectives must declare at least one metric")
    objs = []
    for metric, direction in raw.items():
        if direction not in ("minimize", "maximize"):
            raise StateError(f"objective '{metric}': direction must be 'minimize' or 'maximize'")
        objs.append(Objective(metric=metric, minimize=(direction == "minimize")))
    return objs


def _parse_constraints(raw: list) -> list[Constraint]:
    out = []
    for c in raw:
        if "metric" not in c:
            raise StateError("constraint entries require 'metric'")
        if c.get("lower") is None and c.get("upper") is None:
            raise StateError(f"constraint '{c['metric']}' requires 'lower' and/or 'upper'")
        out.append(Constraint(metric=c["metric"], lower=c.get("lower"), upper=c.get("upper")))
    return out


def summary(frame: Frame) -> dict:
    return {
        "space": frame.space.to_json(),
        "objectives": _objectives_json(frame),
        "constraints": _constraints_json(frame),
        "acqf": frame.shelf.acqf,
        "is_moo": frame.shelf.is_moo,
        "surrogate": frame.shelf.surrogate,
        "region": frame.shelf.region,
        "sampler": frame.shelf.sampler,
        "prior": frame.shelf.prior,
        "default_llm": dict(frame.default_llm or {}),
    }


def _apply_plugin_create_args(frame: Frame, args) -> None:
    for plugin in all_plugins():
        plugin.apply_create_args(frame, args)


# -- commands ---------------------------------------------------------------


def cmd_create(args) -> tuple[Frame, dict]:
    space = SearchSpace.from_json(json.loads(args.space))
    objectives = _parse_objectives(json.loads(args.objectives))
    constraints = _parse_constraints(json.loads(args.constraints) if args.constraints else [])
    acqf = args.acqf or "noisy_logei"
    acqf_params = {"beta": args.beta} if args.beta is not None else {}
    _check_acqf_compat(acqf, is_moo=len(objectives) > 1)

    surrogate = getattr(args, "surrogate", None) or "fixed"
    _check_surrogate_compat(surrogate, is_moo=len(objectives) > 1, has_constraints=bool(constraints))

    shelf = Shelf(
        objectives=objectives,
        constraints=constraints,
        acqf=acqf,
        acqf_params=acqf_params,
        bounds={},
        surrogate=surrogate,
        seed=getattr(args, "seed", None),
        budget=getattr(args, "budget", None),
    )
    frame = Frame(space=space, shelf=shelf)
    default = spec_from_args(args, "llm")
    sidecar = load_sidecar(args.state)
    if not spec_complete(default):
        default = core_spec_from_sidecar(sidecar)
    if spec_complete(default):
        frame.default_llm = default
    _apply_plugin_create_args(frame, args)
    apply_sidecar_plugin_overrides(frame, sidecar)
    frame.log_event("create")
    return frame, summary(frame)


def cmd_suggest(frame: Frame, args) -> tuple[Frame, list[dict]]:
    encoder = Encoder(frame.space)
    bounds_override = json.loads(args.bounds) if args.bounds else None
    around = None
    if args.around is not None:
        around = True if args.around is True else json.loads(args.around)
    result = opt.suggest(
        frame, encoder, q=args.q, bounds_override=bounds_override, around=around, radius=args.radius
    )
    frame.log_event("suggest", q=args.q)
    return frame, result


def cmd_submit(frame: Frame, args) -> tuple[Frame, dict]:
    config = json.loads(args.config)
    metrics = json.loads(args.metrics) if args.metrics else None
    trial = frame.submit(config, metrics)
    frame.log_event("submit", trial_id=trial.trial_id, observed=metrics is not None)
    if metrics is not None:
        for plugin in enabled_plugins(frame):
            plugin.on_observe(frame, trial)
    return frame, {"trial_id": trial.trial_id, "config": trial.config, "status": trial.status}


def cmd_observe(frame: Frame, args) -> tuple[Frame, dict]:
    config = json.loads(args.config)
    metrics = json.loads(args.metrics)
    trial = frame.observe(config, metrics)
    if trial is None:
        outstanding = [t.config for t in frame.in_flight_trials()]
        raise NoMatchingSubmission(outstanding)
    frame.log_event("observe", trial_id=trial.trial_id)
    for plugin in enabled_plugins(frame):
        plugin.on_observe(frame, trial)
    return frame, {"trial_id": trial.trial_id, "config": trial.config, "metrics": trial.metrics}


def cmd_set_bounds(frame: Frame, args) -> tuple[Frame, dict]:
    encoder = Encoder(frame.space)
    new_bounds = json.loads(args.bounds)
    encoder.encode_bounds({**frame.shelf.bounds, **new_bounds})  # validates subset of domain
    frame.shelf.bounds = {**frame.shelf.bounds, **new_bounds}
    frame.log_event("set-bounds", bounds=new_bounds)
    return frame, {"bounds": frame.shelf.bounds, "is_moo": frame.shelf.is_moo, "observed": len(frame.observed_trials())}


def cmd_set_acqf(frame: Frame, args) -> tuple[Frame, dict]:
    _check_acqf_compat(args.acqf, is_moo=frame.shelf.is_moo)
    frame.shelf.acqf = args.acqf
    frame.shelf.acqf_params = {"beta": args.beta} if args.beta is not None else {}
    frame.log_event("set-acqf", acqf=args.acqf)
    return frame, {"acqf": frame.shelf.acqf, "is_moo": frame.shelf.is_moo, "observed": len(frame.observed_trials())}


def cmd_set_objectives(frame: Frame, args) -> tuple[Frame, dict]:
    frame.shelf.objectives = _parse_objectives(json.loads(args.objectives))
    frame.log_event("set-objectives", objectives=_objectives_json(frame))
    return frame, {"is_moo": frame.shelf.is_moo, "objectives": _objectives_json(frame)}


def cmd_set_constraints(frame: Frame, args) -> tuple[Frame, dict]:
    frame.shelf.constraints = _parse_constraints(json.loads(args.constraints))
    frame.log_event("set-constraints", constraints=_constraints_json(frame))
    return frame, {"constraints": _constraints_json(frame), "is_moo": frame.shelf.is_moo}


def cmd_set_surrogate(frame: Frame, args) -> tuple[Frame, dict]:
    _check_surrogate_compat(args.surrogate, is_moo=frame.shelf.is_moo, has_constraints=bool(frame.shelf.constraints))
    frame.shelf.surrogate = args.surrogate
    _apply_plugin_create_args(frame, args)
    frame.log_event("set-surrogate", surrogate=args.surrogate)
    return frame, _slot_summary(frame)


def cmd_set_llm(frame: Frame, args) -> tuple[Frame, dict]:
    spec = spec_from_args(args, "llm")
    if not spec_complete(spec):
        raise StateError("set-llm requires --provider and --model")
    frame.default_llm = spec
    frame.log_event("set-llm", provider=spec["provider"], model=spec["model"])
    return frame, {"default_llm": spec}


def cmd_set_region(frame: Frame, args) -> tuple[Frame, dict]:
    policy = args.policy
    known = ["box"] + [p.name for p in plugins_for_slot("region")]
    if policy not in known:
        raise PluginError(f"unknown region policy '{policy}' (expected {known})")
    frame.shelf.region = policy
    if policy != "box":
        plugin = plugin_by_name(policy)
        plugin.ensure(frame, Encoder(frame.space)) if hasattr(plugin, "ensure") else None
    frame.log_event("set-region", region=policy)
    return frame, {"region": frame.shelf.region, **_slot_summary(frame)}


def cmd_set_sampler(frame: Frame, args) -> tuple[Frame, dict]:
    name = args.sampler
    known = ["botorch"] + [p.name for p in plugins_for_slot("sampler")]
    if name not in known:
        raise PluginError(f"unknown sampler '{name}' (expected {known})")
    frame.shelf.sampler = name
    frame.log_event("set-sampler", sampler=name)
    return frame, {"sampler": frame.shelf.sampler, **_slot_summary(frame)}


def cmd_plugins(frame: Frame, args) -> tuple[None, dict]:
    installed = [
        {
            "name": p.name,
            "slot": p.slot,
            "occupied": occupant(frame, p.slot) is p,
        }
        for p in all_plugins()
    ]
    return None, {
        "slots": {
            "surrogate": frame.shelf.surrogate,
            "region": frame.shelf.region,
            "sampler": frame.shelf.sampler,
            "prior": frame.shelf.prior,
        },
        "plugins": installed,
    }


def _slot_summary(frame: Frame) -> dict:
    out = {
        "surrogate": frame.shelf.surrogate,
        "region": frame.shelf.region,
        "sampler": frame.shelf.sampler,
        "prior": frame.shelf.prior,
    }
    for plugin in enabled_plugins(frame):
        extra = plugin.summary(frame)
        if extra:
            out[plugin.name] = extra
    return out


def cmd_status(frame: Frame, args) -> tuple[None, dict]:
    encoder = Encoder(frame.space)
    return None, {
        "space": frame.space.to_json(),
        "objectives": _objectives_json(frame),
        "constraints": _constraints_json(frame),
        "acqf": frame.shelf.acqf,
        "acqf_params": frame.shelf.acqf_params,
        "bounds": frame.shelf.bounds,
        "is_moo": frame.shelf.is_moo,
        "n_observed": len(frame.observed_trials()),
        "n_in_flight": len(frame.in_flight_trials()),
        "warmup": opt.needs_warmup(frame, encoder),
        "default_llm": dict(frame.default_llm or {}),
        **_slot_summary(frame),
    }


def cmd_diagnostics(frame: Frame, args) -> tuple[None, dict]:
    encoder = Encoder(frame.space)
    model_set = build_model_set(frame, encoder)
    return None, {**compute_diagnostics(model_set, encoder), **_slot_summary(frame)}


def cmd_predict(frame: Frame, args) -> tuple[None, list[dict]]:
    encoder = Encoder(frame.space)
    configs = json.loads(args.configs)
    return None, opt.predict(frame, encoder, configs)


def cmd_score(frame: Frame, args) -> tuple[None, list[dict]]:
    encoder = Encoder(frame.space)
    configs = json.loads(args.configs)
    acqf_names = [a.strip() for a in args.acqf.split(",")]
    return None, opt.score(frame, encoder, configs, acqf_names)


def cmd_trials(frame: Frame, args) -> tuple[None, dict]:
    return None, {
        "trials": [
            {
                "trial_id": t.trial_id,
                "config": t.config,
                "metrics": t.metrics,
                "status": t.status,
            }
            for t in frame.trials
        ]
    }


def cmd_incumbent(frame: Frame, args) -> tuple[None, dict]:
    encoder = Encoder(frame.space)
    trial = opt.get_incumbent(frame, encoder, in_bounds=args.in_bounds)
    if trial is None:
        return None, {"config": None, "metrics": None}
    return None, {"trial_id": trial.trial_id, "config": trial.config, "metrics": trial.metrics}


def cmd_pareto(frame: Frame, args) -> tuple[None, dict]:
    trials = opt.get_pareto(frame)
    return None, {
        "pareto": [{"trial_id": t.trial_id, "config": t.config, "metrics": t.metrics} for t in trials]
    }
