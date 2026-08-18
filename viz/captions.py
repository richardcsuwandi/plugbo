"""One-line experiment captions derived from compare-group directory names."""

from __future__ import annotations

import re

# Longest suffixes first so parsing is unambiguous.
_LAYOUTS: list[tuple[str, str]] = [
    (
        "-noblind-compare-3config-shifted",
        "Identity revealed, shifted optimum: compare vanilla BO, Sara+Lenz, and CAKE on {bench}.",
    ),
    (
        "-noblind-compare-3config",
        "Identity revealed: compare vanilla BO, Sara+Lenz, and CAKE on {bench}.",
    ),
    (
        "-noblind-compare-vanilla",
        "Fixed vanilla BO: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-noblind-compare-cake",
        "Fixed Sara+Lenz+CAKE: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-noblind-compare",
        "Fixed Sara+Lenz: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-compare",
        "Blind search: compare vanilla BO, Sara+Lenz, and CAKE on {bench}.",
    ),
]

_BENCH_LABELS: dict[str, str] = {
    "hartmann6": "Hartmann-6",
    "constrained_hartmann6": "constrained Hartmann-6",
    "branin": "Branin",
    "ackley10": "Ackley-10",
    "ackley20": "Ackley-20",
    "rosenbrock": "Rosenbrock",
    "rastrigin6": "Rastrigin-6",
    "levy": "Levy",
    "griewank": "Griewank",
    "michalewicz": "Michalewicz",
    "styblinski_tang": "Styblinski-Tang",
    "shekel": "Shekel",
    "six_hump_camel": "six-hump camel",
}


def bench_label(benchmark: str) -> str:
    if benchmark in _BENCH_LABELS:
        return _BENCH_LABELS[benchmark]
    m = re.fullmatch(r"gp_sample(\d+)", benchmark)
    if m:
        return f"GP sample ({m.group(1)}D)"
    m = re.fullmatch(r"ackley(\d+)", benchmark)
    if m:
        return f"Ackley-{m.group(1)}"
    return benchmark.replace("_", " ")


def experiment_caption(group_name: str) -> str:
    """Short goal statement for a compare-group directory name."""
    leaf = group_name.rstrip("/").split("/")[-1]
    for suffix, template in _LAYOUTS:
        if leaf.endswith(suffix):
            benchmark = leaf[: -len(suffix)]
            if benchmark:
                return template.format(bench=bench_label(benchmark))
    return f"Compare conditions on {leaf.replace('-', ' ')}."
