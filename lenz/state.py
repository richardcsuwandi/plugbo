"""Persistent state: the trial log, event log, and shelf (live configuration).

`state.json` is the single source of truth. Every `lenz` invocation loads it,
mutates it, and saves it back; reconfiguration (`set-*`) never discards trials.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .space import SearchSpace


class StateError(ValueError):
    pass


@dataclass
class Trial:
    trial_id: str
    config: dict
    metrics: dict | None = None
    status: str = "in_flight"  # "in_flight" | "observed"
    created_at: float = field(default_factory=time.time)
    observed_at: float | None = None
    # Continuous GP-space proposal before int rounding / one-hot argmax.
    # The oracle sees `config` (the projection); the surrogate fits on `x_gp`.
    x_gp: list[float] | None = None


@dataclass
class Objective:
    metric: str
    minimize: bool


@dataclass
class Constraint:
    metric: str
    lower: float | None = None
    upper: float | None = None


def _default_kernel_evolution_state() -> dict:
    return {"generation": 0, "last_evolved_at_n_observed": 0, "frozen": False}


@dataclass
class Shelf:
    objectives: list[Objective] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    acqf: str = "noisy_logei"
    acqf_params: dict = field(default_factory=dict)
    bounds: dict[str, list[float]] = field(default_factory=dict)

    # -- CAKE adaptive-kernel surrogate (see lenz/cake.py) -------------
    surrogate: str = "fixed"  # "fixed" | "cake"
    seed: int | None = None  # pins Sobol warmup / --acqf sobol draws; None = unseeded
    sobol_drawn: int = 0  # how many seeded Sobol points have already been issued
    budget: int | None = None  # total evaluation budget, for the freeze schedule below
    kernel_llm: dict = field(default_factory=dict)  # {"provider", "model", "base_url", "api_key_env", "extra_body"} -- never a raw key
    kernel_population_size: int = 6
    kernel_init_after: int = 6
    kernel_evolve_every: int = 4
    kernel_freeze_fraction: float = 0.5
    kernel_num_crossover: int = 1
    kernel_mutation_prob: float = 0.7
    # Per-metric kernel populations: one CAKE instance per objective metric and
    # per constraint metric. Values: [{"expression", "bic", "generation"}, ...]
    kernel_populations: dict[str, list[dict]] = field(default_factory=dict)
    kernel_evolution_states: dict[str, dict] = field(default_factory=dict)

    @property
    def is_moo(self) -> bool:
        return len(self.objectives) > 1


@dataclass
class Frame:
    space: SearchSpace
    shelf: Shelf
    trials: list[Trial] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    # Latest `suggest` batch: projected config -> continuous proposal, so a
    # later `submit --config` (Sara, harness) can recover `x_gp` without the
    # caller having to pass it.
    pending_x_gp: list[dict] = field(default_factory=list)

    # -- trial log helpers -------------------------------------------------
    def observed_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.status == "observed"]

    def in_flight_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.status == "in_flight"]

    def find_in_flight(self, config: dict) -> Trial | None:
        for t in self.in_flight_trials():
            if configs_match(t.config, config):
                return t
        return None

    def submit(self, config: dict, metrics: dict | None = None, x_gp: list[float] | None = None) -> Trial:
        self.space.validate_config_keys(config)
        if metrics is not None:
            self._check_budget()
        if x_gp is None:
            x_gp = self.take_suggestion_x_gp(config)
        trial = Trial(trial_id=str(uuid.uuid4())[:8], config=config, x_gp=x_gp)
        if metrics is not None:
            trial.metrics = metrics
            trial.status = "observed"
            trial.observed_at = time.time()
        self.trials.append(trial)
        return trial

    def remember_suggestion(self, config: dict, x_gp: list[float] | None) -> None:
        if x_gp is None:
            return
        self.pending_x_gp.append({"config": config, "x_gp": [float(v) for v in x_gp]})

    def clear_pending_x_gp(self) -> None:
        self.pending_x_gp = []

    def take_suggestion_x_gp(self, config: dict) -> list[float] | None:
        for i, item in enumerate(self.pending_x_gp):
            if configs_match(item["config"], config):
                return self.pending_x_gp.pop(i)["x_gp"]
        return None

    def observe(self, config: dict, metrics: dict) -> Trial | None:
        trial = self.find_in_flight(config)
        if trial is None:
            return None
        self._check_budget()
        trial.metrics = metrics
        trial.status = "observed"
        trial.observed_at = time.time()
        return trial

    def _check_budget(self) -> None:
        """Hard cap: once `shelf.budget` observed trials are recorded, refuse
        to record any more (raised as a normal StateError so the CLI reports
        it as `{"ok": false, ...}` instead of crashing the caller). `budget`
        is None unless `lenz create --budget N` was passed, so this is a
        no-op for callers that never opted in (examples, ad-hoc CLI use).
        """
        budget = self.shelf.budget
        if budget is None:
            return
        n_observed = len(self.observed_trials())
        if n_observed >= budget:
            raise StateError(
                f"evaluation budget exhausted ({n_observed}/{budget} observed) -- "
                "no further evaluations may be recorded; report your final incumbent and stop"
            )

    def log_event(self, command: str, **kwargs: Any) -> None:
        self.events.append({"ts": time.time(), "command": command, **kwargs})

    # -- serialization -------------------------------------------------
    def to_json(self) -> dict:
        return {
            "space": self.space.to_json(),
            "shelf": {
                "objectives": [{"metric": o.metric, "minimize": o.minimize} for o in self.shelf.objectives],
                "constraints": [
                    {"metric": c.metric, "lower": c.lower, "upper": c.upper} for c in self.shelf.constraints
                ],
                "acqf": self.shelf.acqf,
                "acqf_params": self.shelf.acqf_params,
                "bounds": self.shelf.bounds,
                "surrogate": self.shelf.surrogate,
                "seed": self.shelf.seed,
                "sobol_drawn": self.shelf.sobol_drawn,
                "budget": self.shelf.budget,
                "kernel_llm": self.shelf.kernel_llm,
                "kernel_population_size": self.shelf.kernel_population_size,
                "kernel_init_after": self.shelf.kernel_init_after,
                "kernel_evolve_every": self.shelf.kernel_evolve_every,
                "kernel_freeze_fraction": self.shelf.kernel_freeze_fraction,
                "kernel_num_crossover": self.shelf.kernel_num_crossover,
                "kernel_mutation_prob": self.shelf.kernel_mutation_prob,
                "kernel_populations": self.shelf.kernel_populations,
                "kernel_evolution_states": self.shelf.kernel_evolution_states,
            },
            "trials": [asdict(t) for t in self.trials],
            "events": self.events,
            "pending_x_gp": self.pending_x_gp,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Frame":
        with open(path) as f:
            obj = json.load(f)
        space = SearchSpace.from_json(obj["space"])
        shelf_obj = obj["shelf"]
        objectives_raw = shelf_obj.get("objectives", [])
        kernel_populations = dict(shelf_obj.get("kernel_populations") or {})
        kernel_evolution_states = dict(shelf_obj.get("kernel_evolution_states") or {})
        # Migrate legacy single-population state.json files.
        legacy_pop = shelf_obj.get("kernel_population") or []
        legacy_state = shelf_obj.get("kernel_evolution_state")
        if legacy_pop and objectives_raw and not kernel_populations:
            primary = objectives_raw[0]["metric"]
            kernel_populations[primary] = legacy_pop
            kernel_evolution_states[primary] = legacy_state or _default_kernel_evolution_state()
        shelf = Shelf(
            objectives=[Objective(**o) for o in objectives_raw],
            constraints=[Constraint(**c) for c in shelf_obj.get("constraints", [])],
            acqf=shelf_obj.get("acqf", "noisy_logei"),
            acqf_params=shelf_obj.get("acqf_params", {}),
            bounds=shelf_obj.get("bounds", {}),
            surrogate=shelf_obj.get("surrogate", "fixed"),
            seed=shelf_obj.get("seed"),
            sobol_drawn=int(shelf_obj.get("sobol_drawn") or 0),
            budget=shelf_obj.get("budget"),
            kernel_llm=shelf_obj.get("kernel_llm", {}),
            kernel_population_size=shelf_obj.get("kernel_population_size", 6),
            kernel_init_after=shelf_obj.get("kernel_init_after", 6),
            kernel_evolve_every=shelf_obj.get("kernel_evolve_every", 4),
            kernel_freeze_fraction=shelf_obj.get("kernel_freeze_fraction", 0.5),
            kernel_num_crossover=shelf_obj.get("kernel_num_crossover", 1),
            kernel_mutation_prob=shelf_obj.get("kernel_mutation_prob", 0.7),
            kernel_populations=kernel_populations,
            kernel_evolution_states=kernel_evolution_states,
        )
        trials = []
        for t in obj.get("trials", []):
            row = dict(t)
            row.setdefault("x_gp", None)
            trials.append(Trial(**row))
        return cls(
            space=space,
            shelf=shelf,
            trials=trials,
            events=obj.get("events", []),
            pending_x_gp=list(obj.get("pending_x_gp") or []),
        )


def configs_match(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not math.isclose(float(va), float(vb), rel_tol=1e-9, abs_tol=1e-9):
                return False
        elif va != vb:
            return False
    return True
