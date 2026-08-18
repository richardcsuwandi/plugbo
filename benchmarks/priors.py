"""Deterministic πBO beliefs used by benchmark experiments.

Fixtures encode only the agent-visible context. They deliberately avoid the
BoLT emulator's hidden best-known coordinates so prior experiments measure
context quality instead of answer leakage.
"""

from __future__ import annotations

import copy


_UNIFORM_TARGET = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}

BOLT_PRIORS: dict[str, dict] = {
    "domain": {
        "lr": {"dist": "beta", "a": 2.0, "b": 4.0},
        "batch": {"dist": "uniform"},
        "lora_rank": {"dist": "normal", "mu": 3.5, "sigma": 1.2},
        "lora_alpha": {"dist": "uniform"},
        "lora_dropout": {"dist": "beta", "a": 2.0, "b": 2.0},
        "lora_layers": {"dist": "beta", "a": 2.0, "b": 1.5},
        "lora_target": {"dist": "categorical", "probs": _UNIFORM_TARGET},
    },
    "generic": {
        "lr": {"dist": "uniform"},
        "batch": {"dist": "uniform"},
        "lora_rank": {"dist": "uniform"},
        "lora_alpha": {"dist": "uniform"},
        "lora_dropout": {"dist": "uniform"},
        "lora_layers": {"dist": "uniform"},
        "lora_target": {"dist": "categorical", "probs": _UNIFORM_TARGET},
    },
    "misleading": {
        "lr": {"dist": "normal", "mu": 0.05, "sigma": 0.04},
        "batch": {"dist": "uniform"},
        "lora_rank": {"dist": "beta", "a": 1.2, "b": 4.0},
        "lora_alpha": {"dist": "beta", "a": 4.0, "b": 1.2},
        "lora_dropout": {"dist": "normal", "mu": 0.05, "sigma": 0.03},
        "lora_layers": {"dist": "beta", "a": 1.2, "b": 4.0},
        "lora_target": {
            "dist": "categorical",
            "probs": {"0": 0.85, "1": 0.05, "2": 0.05, "3": 0.05},
        },
    },
}


def get_prior_fixture(benchmark_name: str, fixture_name: str) -> dict:
    """Return an isolated copy of a named benchmark belief."""
    if benchmark_name != "bolt_lora":
        raise ValueError("prior fixtures are currently defined only for bolt_lora")
    name = fixture_name.removeprefix("bolt-")
    if name not in BOLT_PRIORS:
        choices = ", ".join(f"bolt-{key}" for key in sorted(BOLT_PRIORS))
        raise ValueError(f"unknown prior fixture {fixture_name!r}; choose one of {choices}")
    return copy.deepcopy(BOLT_PRIORS[name])
