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
        "-disclosure-sara-lenz-cake",
        "Fixed Sara+Lenz+CAKE: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-disclosure-sara-only",
        "Fixed Sara-only: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-disclosure-sara-lenz",
        "Fixed Sara+Lenz: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-disclosure-vanilla",
        "Fixed vanilla BO: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-disclosure-cake",
        "Fixed CAKE: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-disclosure-turbo",
        "Fixed TuRBO: compare blind, revealed+shifted, and revealed disclosure on {bench}.",
    ),
    (
        "-revealed-shift",
        "Identity revealed, shifted optimum: compare conditions on {bench}.",
    ),
    (
        "-misleading-compare",
        "Misleading LoRA folklore in context.md: compare conditions on {bench}.",
    ),
    (
        "-generic-compare",
        "Generic context (names and types kept, no LoRA story): compare conditions on {bench}.",
    ),
    (
        "-generic",
        "Generic context (names and types kept, no LoRA story): compare conditions on {bench}.",
    ),
    (
        "-domain",
        "Domain LoRA/Qwen context (real names, no textbook optimum): compare conditions on {bench}.",
    ),
    (
        "-misleading",
        "Misleading LoRA folklore in context.md: compare conditions on {bench}.",
    ),
    (
        "-revealed",
        "Identity revealed: compare conditions on {bench}.",
    ),
    (
        "-blind",
        "Blind search: compare conditions on {bench}.",
    ),
    (
        "-compare",
        "Blind search: compare vanilla BO, Sara+Lenz, and CAKE on {bench}.",
    ),
]

# `*-compare` / `*-blind` is a blind backend sweep for textbook functions.
# bolt_lora-compare and results/logs/bolt_lora are revealed mixed-type HPO.
_REVEALED_COMPARE_BENCHES = frozenset({"bolt_lora"})
_BOLT_DOMAIN_CAPTION = (
    "Domain LoRA/Qwen context (real names, no textbook optimum): compare conditions on {bench}."
)

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
    "bolt_lora": "BOLT LoRA HPO",
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
    if leaf == "bolt_lora":
        return _BOLT_DOMAIN_CAPTION.format(bench=bench_label("bolt_lora"))
    seed_m = re.fullmatch(r"(.+)-seed(\d+)-compare", leaf)
    if seed_m:
        return (
            f"Revealed mixed-type HPO, seed {seed_m.group(2)}: compare conditions on "
            f"{bench_label(seed_m.group(1))}."
        )
    for suffix, template in _LAYOUTS:
        if leaf.endswith(suffix):
            benchmark = leaf[: -len(suffix)]
            if benchmark:
                if suffix == "-compare" and benchmark in _REVEALED_COMPARE_BENCHES:
                    return _BOLT_DOMAIN_CAPTION.format(bench=bench_label(benchmark))
                return template.format(bench=bench_label(benchmark))
    return f"Compare conditions on {leaf.replace('-', ' ')}."


BACKEND_LABELS = {
    "vanilla": "Vanilla",
    "cake": "CAKE",
    "turbo": "TuRBO",
    "pibo": "πBO",
    "llambo": "LLAMBO",
    "cake-turbo": "CAKE+TuRBO",
    "cake-turbo-pibo": "CAKE+TuRBO+πBO",
    "cake-turbo-llambo": "CAKE+TuRBO+LLAMBO",
    "sara-lenz": "Sara+lenz",
    "sara-cake": "Sara+CAKE",
    "sara-only": "Sara-only",
}

DISCLOSURE_LABELS = {
    "blind": "Blind",
    "revealed": "Revealed",
    "shifted": "Shifted",
}

_CONDITION_BACKEND = {
    "vanilla": "vanilla",
    "cake": "cake",
    "turbo": "turbo",
    "pibo": "pibo",
    "llambo": "llambo",
    "cake-turbo": "cake-turbo",
    "cake-turbo-pibo": "cake-turbo-pibo",
    "cake-turbo-llambo": "cake-turbo-llambo",
    "sara-lenz": "sara-lenz",
    "sara-lenz-cake": "sara-cake",
    "sara-only": "sara-only",
}

_CONDITION_DISCLOSURE = {
    "blind": "blind",
    "noblind": "revealed",
    "noblind-shift": "shifted",
    "revealed": "revealed",
    "revealed-shift": "shifted",
}

_GROUP_BACKEND = {
    "-noblind-compare-vanilla": "vanilla",
    "-noblind-compare-cake": "sara-cake",
    "-noblind-compare": "sara-lenz",
}


def parse_group_leaf(leaf: str) -> dict:
    """Benchmark / axis / default disclosure / backend implied by a compare-group folder name."""
    out = {
        "benchmark": None,
        "axis": "other",
        "disclosure": None,
        "backend": None,
        "context": None,
        "seed": None,
    }
    seed_m = re.fullmatch(r"(.+)-seed(\d+)-compare", leaf)
    if seed_m:
        out["benchmark"] = seed_m.group(1)
        out["axis"] = "backend"
        out["disclosure"] = "revealed"
        out["seed"] = int(seed_m.group(2))
        return out
    if leaf == "bolt_lora":
        out["benchmark"] = "bolt_lora"
        out["axis"] = "backend"
        out["disclosure"] = "revealed"
        out["context"] = "domain"
        return out
    for suffix, _template in _LAYOUTS:
        if leaf.endswith(suffix) and len(leaf) > len(suffix):
            out["benchmark"] = leaf[: -len(suffix)]
            if suffix in ("-compare", "-blind"):
                out["axis"] = "backend"
                if out["benchmark"] in _REVEALED_COMPARE_BENCHES:
                    out["disclosure"] = "revealed"
                    out["context"] = "domain"
                else:
                    out["disclosure"] = "blind"
            elif suffix in ("-generic-compare", "-generic"):
                out["axis"] = "backend"
                out["disclosure"] = "revealed"
                out["context"] = "generic"
            elif suffix == "-domain":
                out["axis"] = "backend"
                out["disclosure"] = "revealed"
                out["context"] = "domain"
            elif suffix in ("-misleading-compare", "-misleading"):
                out["axis"] = "backend"
                out["disclosure"] = "revealed"
                out["context"] = "misleading"
            elif suffix in ("-noblind-compare-3config-shifted", "-revealed-shift"):
                out["axis"] = "backend"
                out["disclosure"] = "shifted"
            elif suffix in ("-noblind-compare-3config", "-revealed"):
                out["axis"] = "backend"
                out["disclosure"] = "revealed"
            elif suffix in _GROUP_BACKEND or suffix.startswith("-disclosure-"):
                out["axis"] = "disclosure"
                out["backend"] = _GROUP_BACKEND.get(suffix) or {
                    "-disclosure-vanilla": "vanilla",
                    "-disclosure-cake": "cake",
                    "-disclosure-turbo": "turbo",
                    "-disclosure-sara-lenz": "sara-lenz",
                    "-disclosure-sara-lenz-cake": "sara-cake",
                    "-disclosure-sara-only": "sara-only",
                }.get(suffix)
            break
    return out


def split_run_path(rel: str) -> tuple[str, str]:
    """`(group, condition)` for a run path. Drops a trailing `sandbox_*` folder."""
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    if parts and parts[-1].startswith("sandbox_"):
        parts = parts[:-1]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return "/".join(parts[:-1]), parts[-1]


def classify_relpath(rel: str) -> dict:
    """Stable filter fields derived from a run or group path under results/logs."""
    group, condition = split_run_path(rel)
    leaf = (group or condition or rel).rstrip("/").split("/")[-1]
    parsed = parse_group_leaf(leaf)
    backend = _CONDITION_BACKEND.get(condition) or parsed["backend"]
    disclosure = _CONDITION_DISCLOSURE.get(condition) or parsed["disclosure"]
    benchmark = parsed["benchmark"]
    heading_leaf = group.split("/")[-1] if group else leaf
    return {
        "group": group or None,
        "condition": condition or None,
        "benchmark": benchmark,
        "benchmark_label": bench_label(benchmark) if benchmark else None,
        "backend": backend,
        "backend_label": BACKEND_LABELS.get(backend) if backend else None,
        "disclosure": disclosure,
        "disclosure_label": DISCLOSURE_LABELS.get(disclosure) if disclosure else None,
        "axis": parsed["axis"],
        "heading": _group_heading(heading_leaf, parsed),
    }


def classify_group(name: str) -> dict:
    """Filter fields for a comparison-group directory (no condition folder)."""
    leaf = name.rstrip("/").split("/")[-1]
    parsed = parse_group_leaf(leaf)
    return {
        "group": name,
        "condition": None,
        "benchmark": parsed["benchmark"],
        "benchmark_label": bench_label(parsed["benchmark"]) if parsed["benchmark"] else None,
        "backend": parsed["backend"],
        "backend_label": BACKEND_LABELS.get(parsed["backend"]) if parsed["backend"] else None,
        "disclosure": parsed["disclosure"],
        "disclosure_label": DISCLOSURE_LABELS.get(parsed["disclosure"]) if parsed["disclosure"] else None,
        "axis": parsed["axis"],
        "heading": _group_heading(leaf, parsed),
    }


def _group_heading(leaf: str, parsed: dict, seed: int | None = None) -> str:
    bench = bench_label(parsed["benchmark"]) if parsed["benchmark"] else leaf.replace("-", " ")
    if parsed.get("seed") is not None:
        return f"{bench} · seed {parsed['seed']}"
    if parsed.get("context") == "domain":
        heading = f"{bench} · domain context"
    elif parsed.get("context") == "generic":
        heading = f"{bench} · generic context"
    elif parsed.get("context") == "misleading":
        heading = f"{bench} · misleading prior"
    elif parsed["axis"] == "disclosure":
        heading = f"{bench} · disclosure"
    elif parsed["disclosure"] == "shifted":
        heading = f"{bench} · revealed+shifted"
    elif parsed["disclosure"] == "revealed":
        heading = f"{bench} · revealed"
    elif parsed["disclosure"] == "blind":
        heading = f"{bench} · blind"
    else:
        heading = bench
    if seed is not None:
        return f"{heading} · seed {seed}"
    return heading


def heading_with_seed(group_name: str, seed: int | None) -> str:
    """Sidebar title for a compare group, with seed when sandboxes record one."""
    leaf = group_name.rstrip("/").split("/")[-1]
    parsed = parse_group_leaf(leaf)
    return _group_heading(leaf, parsed, seed=seed)
