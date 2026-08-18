"""Applies the paper's anti-memorization transform to a benchmark: renamed
parameters, a unit-hypercube domain, and a randomly shifted optimum -- so an
agent cannot submit a memorized textbook optimum directly and must actually
search. ("On the experimental setup"; the sandbox itself, built in
`sandbox.py`, additionally hides the benchmark's identity from the filesystem.)

`build_obfuscated(..., reveal=True)` builds the deliberate *opposite*
transform for `run_noblind_test.py`: real parameter names, the benchmark's
true textbook bounds exposed directly (instead of a [0, 1] cube), and --
unless `shift=True` is also passed -- no relocation of the optimum at all.
This is how we test whether a model can one-shot a benchmark purely from
memorized knowledge of its name, once that name is no longer hidden.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from .functions import get_spec

MAX_SHIFT_FRACTION = 0.25  # paper: "relocated ... by up to a quarter of every dimension's range"


def _random_param_names(dim: int, rng: random.Random) -> list[str]:
    alphabet = string.ascii_lowercase
    names: set[str] = set()
    while len(names) < dim:
        names.add("".join(rng.choices(alphabet, k=4)))
    ordered = list(names)
    rng.shuffle(ordered)
    return ordered


@dataclass
class ObfuscatedBenchmark:
    spec: BenchmarkSpec  # the true identity -- never written into a *blind* sandbox
    param_names: list[str]
    shift_frac: list[float]  # per-dim shift in [-0.25, 0.25], applied as a torus wraparound
    seed: int  # persisted so gp_sample's spec (a function of seed) can be reconstructed at scoring time
    reveal_bounds: bool = False  # True: agent submits/receives values in true units, not a [0, 1] cube
    # Agent-facing choice labels per param, same order as the true `values` list.
    # Empty for continuous-only specs.
    choice_values: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.spec.dim

    def unit_to_true(self, config: dict) -> list:
        if self.spec.space is not None:
            return self._mixed_to_true(config)
        x_true = []
        for i, name in enumerate(self.param_names):
            v = float(config[name])
            lower, upper = self.spec.bounds[i]
            u = (v - lower) / (upper - lower) if self.reveal_bounds else v
            wrapped = (u + self.shift_frac[i]) % 1.0
            x_true.append(lower + wrapped * (upper - lower))
        return x_true

    def _mixed_to_true(self, config: dict) -> list:
        """Native mixed-type decode. Values are already in true units (no unit-cube,
        no torus shift): renaming is the only blind transform.
        """
        x_true = []
        for i, dim in enumerate(self.spec.space.values()):
            name = self.param_names[i]
            raw = config[name]
            if dim["kind"] == "choice":
                agent_vals = self.choice_values.get(name, dim["values"])
                if raw not in agent_vals and isinstance(raw, float) and raw == int(raw):
                    raw = int(raw)
                idx = agent_vals.index(raw)
                x_true.append(dim["values"][idx])
            elif dim.get("type") == "int":
                x_true.append(int(round(float(raw))))
            else:
                x_true.append(float(raw))
        return x_true

    def evaluate(self, config: dict) -> float:
        return self.spec.fn(self.unit_to_true(config))

    def evaluate_metrics(self, config: dict) -> dict:
        """Like `evaluate`, but also includes the constraint metric ("c")
        when the benchmark has one -- what the generated oracle actually
        prints (see `sandbox.py`)."""
        x_true = self.unit_to_true(config)
        metrics = {"y": self.spec.fn(x_true)}
        if self.spec.constraint_fn is not None:
            metrics["c"] = self.spec.constraint_fn(x_true)
        return metrics

    def true_to_unit(self, x_true: list[float]) -> dict:
        """Inverse of `unit_to_true`: the agent-facing config that maps to a
        given true-coordinate point. Used for scoring/testing against the
        benchmark's known optimum location, never exposed to the agent."""
        if self.spec.space is not None:
            return self._true_to_mixed(x_true)
        config = {}
        for i, name in enumerate(self.param_names):
            lower, upper = self.spec.bounds[i]
            wrapped = (x_true[i] - lower) / (upper - lower)
            u = (wrapped - self.shift_frac[i]) % 1.0
            config[name] = lower + u * (upper - lower) if self.reveal_bounds else u
        return config

    def _true_to_mixed(self, x_true: list) -> dict:
        config = {}
        for i, dim in enumerate(self.spec.space.values()):
            name = self.param_names[i]
            val = x_true[i]
            if dim["kind"] == "choice":
                if val not in dim["values"] and isinstance(val, float) and val == int(val):
                    val = int(val)
                idx = dim["values"].index(val)
                agent_vals = self.choice_values.get(name, dim["values"])
                config[name] = agent_vals[idx]
            elif dim.get("type") == "int":
                config[name] = int(round(float(val)))
            else:
                config[name] = float(val)
        return config

    def unit_space_json(self) -> dict:
        """The agent-facing search space. Blind mode: every dimension is a
        plain [0, 1] range under its randomized name -- no hint of the
        original domain. Reveal mode: the benchmark's real bounds, under
        real names -- see module docstring.

        Mixed-type specs (`spec.space`) keep native kinds (int / choice) in
        both modes; only names (and choice labels) are obfuscated when blind.
        """
        if self.spec.space is not None:
            out = {}
            for i, dim in enumerate(self.spec.space.values()):
                name = self.param_names[i]
                entry = dict(dim)
                if dim["kind"] == "choice":
                    entry = {**entry, "values": list(self.choice_values.get(name, dim["values"]))}
                out[name] = entry
            return out
        if self.reveal_bounds:
            return {
                name: {"kind": "range", "lower": lo, "upper": hi}
                for name, (lo, hi) in zip(self.param_names, self.spec.bounds)
            }
        return {name: {"kind": "range", "lower": 0.0, "upper": 1.0} for name in self.param_names}

    def objectives_json(self) -> dict:
        return {"y": "minimize" if self.spec.minimize else "maximize"}

    def to_secret(self) -> dict:
        """Everything needed to reconstruct this transform for scoring, kept
        OUT of the sandbox the agent can see."""
        secret = {
            "benchmark": self.spec.name,
            "param_names": self.param_names,
            "shift_frac": self.shift_frac,
            "seed": self.seed,
            "reveal_bounds": self.reveal_bounds,
        }
        if self.choice_values:
            secret["choice_values"] = self.choice_values
        return secret

    @classmethod
    def from_secret(cls, secret: dict) -> "ObfuscatedBenchmark":
        seed = secret.get("seed")
        return cls(
            spec=get_spec(secret["benchmark"], seed),
            param_names=list(secret["param_names"]),
            shift_frac=list(secret["shift_frac"]),
            seed=seed,
            reveal_bounds=secret.get("reveal_bounds", False),
            choice_values=dict(secret.get("choice_values") or {}),
        )


def _identity_param_names(dim: int) -> list[str]:
    return [f"x{i + 1}" for i in range(dim)]


def build_obfuscated(benchmark_name: str, seed: int, reveal: bool = False, shift: bool = True) -> ObfuscatedBenchmark:
    """`reveal=True` builds the no-blind transform for `run_noblind_test.py`:
    real param names, true bounds exposed directly, and (unless `shift=True`
    is explicitly requested too) no relocation -- so a model that has the
    benchmark's textbook optimum memorized can submit it verbatim, in its
    original units, on the very first evaluation. `reveal=False` (default)
    is the paper's blind anti-memorization transform, unchanged.
    """
    spec = get_spec(benchmark_name, seed)
    rng = random.Random(seed)
    n_choice_labels = 0
    if spec.space is not None and not reveal:
        n_choice_labels = sum(len(d["values"]) for d in spec.space.values() if d["kind"] == "choice")
    extra_labels: list[str] = []
    if reveal:
        param_names = list(spec.space.keys()) if spec.space is not None else _identity_param_names(spec.dim)
    else:
        pool = _random_param_names(spec.dim + n_choice_labels, rng)
        param_names = pool[: spec.dim]
        extra_labels = pool[spec.dim :]
    use_shift = (shift if reveal else True) and spec.allow_shift
    shift_frac = [rng.uniform(-MAX_SHIFT_FRACTION, MAX_SHIFT_FRACTION) for _ in range(spec.dim)] if use_shift else [0.0] * spec.dim
    choice_values: dict = {}
    if spec.space is not None:
        li = 0
        for i, dim in enumerate(spec.space.values()):
            if dim["kind"] != "choice":
                continue
            name = param_names[i]
            if reveal:
                choice_values[name] = list(dim["values"])
            else:
                k = len(dim["values"])
                choice_values[name] = extra_labels[li : li + k]
                li += k
    return ObfuscatedBenchmark(
        spec=spec,
        param_names=param_names,
        shift_frac=shift_frac,
        seed=seed,
        reveal_bounds=reveal,
        choice_values=choice_values,
    )
