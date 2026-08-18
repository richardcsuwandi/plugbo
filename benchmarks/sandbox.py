"""Builds a blind-test sandbox mirroring the paper's benchmark-recognition
countermeasures ("On the experimental setup"): a randomly-named
directory containing only a generic task description, a random token, and a
symlink to the evaluation oracle -- no benchmark name in the domain, the
parameter names, the path, or the directory contents. The oracle's own source
is generated from hand-written generic formula templates below (not by
introspecting `functions.py`) so its identifying function/variable names never
appear in a file the agent's `bash` tool could read.

The true identity (benchmark name, parameter mapping, optimum shift) is
written to a `secret.json` OUTSIDE the sandbox, for scoring the run
afterward without ever exposing it to the agent.

`build_sandbox(..., reveal=True)` builds the opposite: a sandbox whose
context.md names the benchmark outright and (unless `shift=True`) whose
optimum sits exactly where the textbook says it does, in real units under
real parameter names. See `benchmarks/run_noblind_test.py`.
"""

from __future__ import annotations

import json
import math
import secrets
import stat
from pathlib import Path

from .functions import BenchmarkSpec
from .obfuscate import ObfuscatedBenchmark, build_obfuscated

_ACKLEY_FORMULA = """\
    a, b, c = 20.0, 0.2, 2 * math.pi
    d = len(x)
    sum_sq = sum(xi ** 2 for xi in x)
    sum_cos = sum(math.cos(c * xi) for xi in x)
    return -a * math.exp(-b * math.sqrt(sum_sq / d)) - math.exp(sum_cos / d) + a + math.e
"""

_BRANIN_FORMULA = """\
    x1, x2 = x
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi, 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
"""

_HARTMANN6_FORMULA = """\
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
    P = [[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]]
    total = 0.0
    for i in range(4):
        inner = sum(A[i][j] * (x[j] - 1e-4 * P[i][j]) ** 2 for j in range(6))
        total += alpha[i] * math.exp(-inner)
    return -total
"""

_HARTMANN6_BALL_CONSTRAINT_FORMULA = """\
    center, r = 0.5, 0.4
    return sum((xi - center) ** 2 for xi in x) - r ** 2
"""

_ROSENBROCK_FORMULA = """\
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return total
"""

_RASTRIGIN_FORMULA = """\
    d = len(x)
    return 10.0 * d + sum(xi ** 2 - 10.0 * math.cos(2 * math.pi * xi) for xi in x)
"""

_LEVY_FORMULA = """\
    w = [1.0 + (xi - 1.0) / 4.0 for xi in x]
    total = math.sin(math.pi * w[0]) ** 2
    for wi in w[:-1]:
        total += (wi - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * wi + 1.0) ** 2)
    total += (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2 * math.pi * w[-1]) ** 2)
    return total
"""

_GRIEWANK_FORMULA = """\
    sum_sq = sum(xi ** 2 for xi in x) / 4000.0
    prod_cos = 1.0
    for i, xi in enumerate(x):
        prod_cos *= math.cos(xi / math.sqrt(i + 1))
    return sum_sq - prod_cos + 1.0
"""

_MICHALEWICZ_FORMULA = """\
    m = 10
    return -sum(math.sin(xi) * math.sin((i + 1) * xi ** 2 / math.pi) ** (2 * m) for i, xi in enumerate(x))
"""

_STYBLINSKI_TANG_FORMULA = """\
    return 0.5 * sum(xi ** 4 - 16.0 * xi ** 2 + 5.0 * xi for xi in x)
"""

_SHEKEL_FORMULA = """\
    beta = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
    C = [[4, 1, 8, 6, 3, 2, 5, 8, 6, 7], [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
         [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]]
    total = 0.0
    for i in range(10):
        inner = sum((x[j] - C[j][i]) ** 2 for j in range(4))
        total += 1.0 / (inner + beta[i] / 10.0)
    return -total
"""

_SIX_HUMP_CAMEL_FORMULA = """\
    x1, x2 = x
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
"""

_FORMULAS = {
    "branin": _BRANIN_FORMULA,
    "hartmann6": _HARTMANN6_FORMULA,
    "constrained_hartmann6": _HARTMANN6_FORMULA,
    "shekel": _SHEKEL_FORMULA,
    "six_hump_camel": _SIX_HUMP_CAMEL_FORMULA,
}

# (name prefix, formula) for the dimension-general families -- checked in
# order after the exact-name lookup above, so e.g. "styblinski_tang6" matches
# the "styblinski_tang" prefix.
_PREFIX_FORMULAS = [
    ("ackley", _ACKLEY_FORMULA),
    ("rosenbrock", _ROSENBROCK_FORMULA),
    ("rastrigin", _RASTRIGIN_FORMULA),
    ("levy", _LEVY_FORMULA),
    ("griewank", _GRIEWANK_FORMULA),
    ("michalewicz", _MICHALEWICZ_FORMULA),
    ("styblinski_tang", _STYBLINSKI_TANG_FORMULA),
]


def _gp_sample_formula(spec: BenchmarkSpec) -> str:
    """Unlike the textbook functions above, there's nothing to hand-obscure
    here -- a GP sample has no name to hide. We just need the oracle to
    reproduce the exact same random Fourier feature sum as `functions.py`
    (see `_gp_sample_spec`), so its weights are embedded as literals.
    """
    w, theta, tau = spec.extra["w"], spec.extra["theta"], spec.extra["tau"]
    scale = math.sqrt(2.0 / len(w))
    return (
        f"    w = {w!r}\n"
        f"    theta = {theta!r}\n"
        f"    tau = {tau!r}\n"
        f"    scale = {scale!r}\n"
        "    total = 0.0\n"
        "    for wi, ti, taui in zip(w, theta, tau):\n"
        "        dot = sum(t * xi for t, xi in zip(ti, x))\n"
        "        total += wi * math.cos(dot + taui)\n"
        "    return scale * total\n"
    )


def _formula_for(spec: BenchmarkSpec) -> str:
    if spec.extra.get("oracle") == "bolt_lora":
        raise KeyError("bolt_lora uses a surrogate oracle, not a formula template")
    if spec.name in _FORMULAS:
        return _FORMULAS[spec.name]
    if spec.name.startswith("gp_sample"):
        return _gp_sample_formula(spec)
    for prefix, formula in _PREFIX_FORMULAS:
        if spec.name.startswith(prefix):
            return formula
    raise KeyError(f"no oracle formula template for benchmark '{spec.name}'")


def _constraint_formula_for(spec: BenchmarkSpec) -> str | None:
    if spec.constraint_fn is None:
        return None
    if spec.name == "constrained_hartmann6":
        return _HARTMANN6_BALL_CONSTRAINT_FORMULA
    raise KeyError(f"no oracle constraint formula template for benchmark '{spec.name}'")


_ORACLE_SOURCE = '''#!/usr/bin/env python3
import json, math, sys

_PARAM_NAMES = {param_names!r}
_BOUNDS = {bounds!r}
_SHIFT = {shift!r}
_REVEAL = {reveal!r}  # True: submitted config values are already in true units, not a [0, 1] cube


def _f(x):
{formula}

{constraint_def}
def main():
    config = json.loads(sys.argv[1])
    x_raw = [float(config[name]) for name in _PARAM_NAMES]
    x_true = []
    for xi, (lo, hi), s in zip(x_raw, _BOUNDS, _SHIFT):
        u = (xi - lo) / (hi - lo) if _REVEAL else xi
        w = (u + s) % 1.0
        x_true.append(lo + w * (hi - lo))
    result = {{"y": _f(x_true)}}
{constraint_call}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
'''


def _verb(ob: ObfuscatedBenchmark) -> str:
    return "Minimize" if ob.spec.minimize else "Maximize"


def _dim_line(name: str, dim: dict) -> str:
    if dim["kind"] == "choice":
        vals = ", ".join(repr(v) for v in dim["values"])
        return f"- `{name}`: categorical in {{{vals}}}"
    lo, hi = dim["lower"], dim["upper"]
    if dim.get("type") == "int":
        return f"- `{name}`: integer in [{lo}, {hi}]"
    return f"- `{name}`: float in [{lo}, {hi}]"


def _blind_context(ob: ObfuscatedBenchmark, constraint_note: str) -> str:
    if ob.spec.space is not None:
        return _mixed_blind_context(ob, constraint_note)
    dims = "\n".join(f"- `{name}`: float in [0, 1]" for name in ob.param_names)
    return (
        "# Black-box optimization task\n\n"
        f"Minimize the scalar output `y` of an unknown function over {ob.dim} continuous "
        "parameters, each already normalized to [0, 1]. No further structure, symmetry, or "
        "prior knowledge is available -- treat this as a genuine black box.\n\n"
        "## Parameters\n\n"
        f"{dims}\n\n"
        "## Evaluation\n\n"
        "Run `./oracle '<config-json>'` (a symlink in this directory); it prints "
        f'`{{"y": <value>{constraint_note}}}`. Minimize `y`.\n\n'
        "## Rules\n\n"
        "Do not read the oracle's own source, and do not write code to fit, regress, "
        "curve-fit, or otherwise numerically infer the shape of `_f`. That defeats the "
        "point of the exercise, which is to optimize a genuine black box under a limited "
        "evaluation budget -- not to reconstruct it analytically. Use only `lenz` and "
        "`./oracle` to search; every evaluation must go through `./oracle`.\n"
    )


def _mixed_blind_context(ob: ObfuscatedBenchmark, constraint_note: str) -> str:
    space = ob.unit_space_json()
    dims = "\n".join(_dim_line(name, dim) for name, dim in space.items())
    verb = _verb(ob)
    kind = "mixed-type"
    return (
        "# Black-box optimization task\n\n"
        f"{verb} the scalar output `y` of an unknown function over {ob.dim} {kind} "
        "parameters. No domain names, model family, or other prior knowledge is available "
        "-- treat this as a genuine black box.\n\n"
        "## Parameters\n\n"
        f"{dims}\n\n"
        "## Evaluation\n\n"
        "Run `./oracle '<config-json>'` (a symlink in this directory); it prints "
        f'`{{"y": <value>{constraint_note}}}`. {verb} `y`.\n\n'
        "## Rules\n\n"
        "Do not read the oracle's own source, and do not write code to fit, regress, "
        "curve-fit, or otherwise numerically infer the shape of `_f`. That defeats the "
        "point of the exercise, which is to optimize a genuine black box under a limited "
        "evaluation budget -- not to reconstruct it analytically. Use only `lenz` and "
        "`./oracle` to search; every evaluation must go through `./oracle`.\n"
    )


_Y_C_SUFFIX = ', "c": <value>'


BOLT_CONTEXT_VARIANTS = ("domain", "generic", "misleading")


def _reveal_context(
    ob: ObfuscatedBenchmark,
    benchmark_name: str,
    shifted: bool,
    context_variant: str = "domain",
) -> str:
    if ob.spec.name == "bolt_lora":
        return _bolt_lora_reveal_context(ob, variant=context_variant)
    dims = "\n".join(f"- `{name}`: float in [{lo}, {hi}]" for name, (lo, hi) in zip(ob.param_names, ob.spec.bounds))
    shift_note = (
        "The optimum has, however, been relocated from its textbook location by a hidden "
        "per-dimension shift (up to a quarter of each dimension's range, wraparound at the "
        "bounds) -- your recalled optimum is a strong *starting point*, not the answer."
        if shifted
        else "The optimum is at its exact, unmodified textbook location -- nothing has been "
        "renamed, rescaled, or shifted."
    )
    constraint_note = ""
    if ob.spec.constraint_fn is not None:
        constraint_note = (
            f'\n\nThis is a **constrained** variant: `./oracle` also reports `"c"`, and a '
            f"configuration is feasible iff `c <= {ob.spec.constraint_upper}`. The unconstrained "
            "textbook optimum is infeasible here by construction -- recalling it tells you "
            "where *not* to look, not where to submit."
        )
    return (
        "# Black-box optimization task -- identity revealed (memorization probe)\n\n"
        f"This is the standard **{benchmark_name}** synthetic test function from the Bayesian "
        f"optimization literature, over its real {ob.dim}-D domain, in its standard textbook "
        f"parameterization (no renamed parameters). {shift_note}{constraint_note}\n\n"
        "If you recognize this function and recall its known global optimum, you are "
        "encouraged to submit that as (or very close to) your very first evaluation -- that is "
        "precisely what this run is measuring: how much of your performance on 'novel' "
        "black-box tasks is actually optimum recall rather than search.\n\n"
        "## Parameters\n\n"
        f"{dims}\n\n"
        "## Evaluation\n\n"
        "Run `./oracle '<config-json>'` (a symlink in this directory); it prints "
        f'`{{"y": <value>{_Y_C_SUFFIX if ob.spec.constraint_fn is not None else ""}}}`. '
        "Minimize `y`" + (" among feasible configurations" if ob.spec.constraint_fn is not None else "") + ".\n\n"
        "## Rules\n\n"
        "Do not read the oracle's own source. Beyond that, anything goes, including directly "
        "recalling or looking up (from your own training, not by executing a search) the "
        "function's known optimum -- that is the point of this probe. Every evaluation must "
        "still go through `./oracle`.\n"
    )


def _bolt_lora_param_block(ob: ObfuscatedBenchmark) -> str:
    space = ob.unit_space_json()
    return "\n".join(_dim_line(name, dim) for name, dim in space.items())


def _bolt_lora_eval_rules() -> str:
    return (
        "## Evaluation\n\n"
        "Run `./oracle '<config-json>'` (a symlink in this directory); it prints "
        '`{"y": <value>}`. Maximize `y`.\n\n'
        "## Rules\n\n"
        "Do not read the oracle's own source. Use only `lenz` and `./oracle` to search; "
        "every evaluation must go through `./oracle`.\n"
    )


def _bolt_lora_reveal_context(ob: ObfuscatedBenchmark, variant: str = "domain") -> str:
    """Agent-visible prompt for revealed bolt_lora. `variant` changes the story,
    not the mixed-type space: domain (LoRA/Qwen), generic (names/types only),
    or misleading (false LoRA folklore). Must never mention f_opt / X_OPT.
    """
    if variant not in BOLT_CONTEXT_VARIANTS:
        raise ValueError(f"unknown bolt_lora context variant {variant!r}; want one of {BOLT_CONTEXT_VARIANTS}")
    dims = _bolt_lora_param_block(ob)
    tail = _bolt_lora_eval_rules()
    if variant == "generic":
        return (
            "# Mixed-type black-box optimization\n\n"
            "Maximize the scalar objective `y` over a 7-dimensional mixed-type search "
            "space. Parameter names, types, and bounds are listed below. There is no "
            "published closed-form optimum to look up, and no other domain knowledge "
            "is provided.\n\n"
            "## Parameters\n\n"
            f"{dims}\n\n"
            f"{tail}"
        )
    if variant == "misleading":
        return (
            "# LoRA hyperparameter optimization\n\n"
            "You are tuning LoRA adapters for a Qwen-class 8B language model. The "
            "objective `y` is a validation score in roughly [0, 1] (higher is better).\n\n"
            "## Established settings\n\n"
            "Treat the following as known-good defaults for this task and start there:\n"
            "- `lora_dropout` near 0.05 (typical LoRA default)\n"
            "- `lora_target` = 0\n"
            "- `lr` small, around 0.05 or below\n"
            "- prefer fewer `lora_layers` rather than adapting every layer\n"
            "- prefer higher `lora_alpha` relative to `lora_rank`\n\n"
            "These are believed to be near-optimal.\n\n"
            "## Parameters\n\n"
            f"{dims}\n\n"
            f"{tail}"
        )
    return (
        "# LoRA hyperparameter optimization\n\n"
        "You are tuning LoRA adapters for a Qwen-class 8B language model. The objective "
        "`y` is a validation score in roughly [0, 1] from an offline emulator of real "
        "training runs (higher is better). There is no published closed-form optimum to "
        "look up; parameter names and scales are real, but numeric best-known coordinates "
        "are not provided.\n\n"
        "## Parameters\n\n"
        f"{dims}\n\n"
        "- `lr`: learning rate, already scaled to [0, 1]\n"
        "- `batch`: mini-batch size (integer)\n"
        "- `lora_rank`: LoRA rank (integer)\n"
        "- `lora_alpha`: LoRA alpha (integer)\n"
        "- `lora_dropout`: LoRA dropout\n"
        "- `lora_layers`: number of layers that receive LoRA (integer)\n"
        "- `lora_target`: which attention projection is adapted (categorical index)\n\n"
        f"{tail}"
    )


_ORACLE_SURROGATE_SOURCE = '''#!/usr/bin/env python3
import json, sys

_PAYLOAD = {payload!r}


def main():
    from benchmarks.bolt_lora import evaluate_payload
    config = json.loads(sys.argv[1])
    print(json.dumps(evaluate_payload(_PAYLOAD, config)))


if __name__ == "__main__":
    main()
'''


def _write_oracle(oracle_path: Path, ob: ObfuscatedBenchmark) -> None:
    if ob.spec.extra.get("oracle") == "bolt_lora":
        from .bolt_lora import oracle_payload

        oracle_path.write_text(
            _ORACLE_SURROGATE_SOURCE.format(payload=oracle_payload(ob.param_names, ob.choice_values, ob.spec.space))
        )
        return
    constraint_formula = _constraint_formula_for(ob.spec)
    oracle_path.write_text(
        _ORACLE_SOURCE.format(
            param_names=ob.param_names,
            bounds=ob.spec.bounds,
            shift=ob.shift_frac,
            reveal=ob.reveal_bounds,
            formula=_formula_for(ob.spec),
            constraint_def="" if constraint_formula is None else f"def _c(x):\n{constraint_formula}\n",
            constraint_call="" if constraint_formula is None else '    result["c"] = _c(x_true)\n',
        )
    )


def build_sandbox(
    benchmark_name: str,
    root: Path,
    seed: int | None = None,
    reveal: bool = False,
    shift: bool = False,
    context_variant: str = "domain",
) -> dict:
    """Creates a randomly-named sandbox dir with a generic context.md, a
    random token, and a symlinked oracle -- and a separate secret record
    (outside the sandbox) for scoring the run afterward.

    `reveal=True` (see `run_noblind_test.py`) builds the opposite of the
    blind anti-memorization sandbox: real parameter names, real bounds, and
    the benchmark's name stated outright in context.md. `shift` only matters
    when `reveal=True` (blind mode always shifts): False (default) leaves the
    optimum at its exact textbook location -- the purest one-shot-recall
    test; True additionally applies the paper's random relocation, so the
    model knows what it's solving but still has to search for the moved
    optimum.

    Returns {"sandbox": Path, "secret_path": Path, "token": str, "constraints": list[dict] | None}.
    """
    seed = seed if seed is not None else secrets.randbits(32)
    if context_variant != "domain" and benchmark_name != "bolt_lora":
        raise ValueError(
            f"context_variant={context_variant!r} is only implemented for bolt_lora"
        )
    ob = build_obfuscated(benchmark_name, seed, reveal=reveal, shift=shift)

    # Resolve before deriving any paths -- `oracle_path` ends up as a symlink
    # target inside `sandbox`, and a relative target is resolved by the OS
    # relative to the symlink's own directory, not the caller's cwd. A
    # relative --root here would silently produce a dangling symlink.
    root = Path(root).resolve()
    answers_dir = root / "_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    token = secrets.token_hex(8)
    sandbox = root / f"sandbox_{token}"
    sandbox.mkdir(parents=True, exist_ok=False)

    oracle_path = answers_dir / f"{token}_oracle.py"
    _write_oracle(oracle_path, ob)
    oracle_path.chmod(oracle_path.stat().st_mode | stat.S_IEXEC)

    secret_path = answers_dir / f"{token}.json"
    secret_path.write_text(json.dumps(ob.to_secret()))

    # No `token.txt` in the sandbox: nothing in this repo reads it back
    # (plot_compare.py derives the token from the `sandbox_<token>` dirname
    # instead), and it would hand the agent's unrestricted `bash` tool the
    # exact filename of the un-obfuscated answer key next door in
    # `_answers/`. `return`ing the token below is enough for callers that
    # need it.
    (sandbox / "oracle").symlink_to(oracle_path)

    constraints = None
    constraint_note = ""
    if ob.spec.constraint_fn is not None:
        constraints = [{"metric": "c", "upper": ob.spec.constraint_upper}]
        constraint_note = ', "c": <value>'

    if reveal:
        context = _reveal_context(
            ob, benchmark_name, shifted=shift, context_variant=context_variant
        )
    else:
        context = _blind_context(ob, constraint_note)
    (sandbox / "context.md").write_text(context)

    return {"sandbox": sandbox, "secret_path": secret_path, "token": token, "constraints": constraints}
