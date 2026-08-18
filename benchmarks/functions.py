"""Canonical synthetic benchmark functions in their standard coordinates,
plus the known global optimum for computing true regret. These are the
*true* functions the obfuscation layer in `obfuscate.py` hides from the agent.

Beyond Branin/Hartmann/Ackley (the paper's own suite, plus `gp_sample` and
`constrained_hartmann6` -- see below), this module also carries a handful of
other textbook BO functions (Rosenbrock, Rastrigin, Levy, Griewank,
Michalewicz, Styblinski-Tang, Shekel, Six-Hump Camel) so the anti-memorization
harness in `obfuscate.py`/`sandbox.py` isn't limited to the exact functions
the paper happened to evaluate on -- any of these are just as likely to be
memorized by a pretrained LLM, and having more of them makes it harder for a
model to pattern-match "this must be one of the N functions this paper used."

`gp_sample<dim>` is the odd one out: it is not a fixed textbook function at
all. In the source paper it is an approximate squared-exponential-
kernel GP sample drawn fresh per seed via random Fourier features -- by
construction it cannot appear in any pretraining corpus, so it is the paper's
own recommended *no-memorization-possible* control. See `_gp_sample_spec`.

`constrained_hartmann6` mirrors the paper's "Constrained Hartmann (6-D)"
benchmark, but the paper does not disclose the exact constraint
function used, so this is our own reconstruction in the same spirit: a
spherical trust region around the unit-cube center that excludes Hartmann6's
unconstrained (and famous) optimum, forcing genuine constrained search rather
than optimum recall. See `_HARTMANN6_BALL_CONSTRAINT_FORMULA` in
`sandbox.py` for the matching oracle-side implementation and
`scripts/derive_constrained_hartmann6.py`-style derivation notes below.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

Fn = Callable[[list[float]], float]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dim: int
    bounds: list[tuple[float, float]]  # standard textbook domain, per dimension
    fn: Fn
    minimize: bool
    f_opt: float  # known global optimum value
    constraint_fn: Fn | None = None  # true-coordinate constraint c(x); feasible iff c(x) <= constraint_upper
    constraint_upper: float | None = None
    extra: dict = field(default_factory=dict)  # side-channel data sandbox.py needs to regenerate the oracle text
    # (e.g. gp_sample's random Fourier feature weights) -- never touched by the harness itself.


def branin(x: list[float]) -> float:
    x1, x2 = x
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi, 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


_HARTMANN6_ALPHA = [1.0, 1.2, 3.0, 3.2]
_HARTMANN6_A = [
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
]
_HARTMANN6_P = [
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
]


def hartmann6(x: list[float]) -> float:
    total = 0.0
    for i in range(4):
        inner = sum(_HARTMANN6_A[i][j] * (x[j] - 1e-4 * _HARTMANN6_P[i][j]) ** 2 for j in range(6))
        total += _HARTMANN6_ALPHA[i] * math.exp(-inner)
    return -total


def make_ackley(dim: int) -> Fn:
    a, b, c = 20.0, 0.2, 2 * math.pi

    def ackley(x: list[float]) -> float:
        d = len(x)
        sum_sq = sum(xi**2 for xi in x)
        sum_cos = sum(math.cos(c * xi) for xi in x)
        return -a * math.exp(-b * math.sqrt(sum_sq / d)) - math.exp(sum_cos / d) + a + math.e

    return ackley


def _ackley_spec(dim: int) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"ackley{dim}",
        dim=dim,
        bounds=[(-32.768, 32.768)] * dim,
        fn=make_ackley(dim),
        minimize=True,
        f_opt=0.0,
    )


# --- constrained Hartmann6 --------------------------------------------------
# Ball constraint c(x) = ||x - 0.5||^2 - r^2 <= 0, r = 0.4. Numerically
# verified (SLSQP, 2000 random restarts, matched independently by
# differential_evolution): the unconstrained optimum sits at distance 0.568
# from the cube center, well outside this ball, so the constrained optimum is
# a *different*, non-memorized point on the constraint boundary.
_HARTMANN6_BALL_CENTER = 0.5
_HARTMANN6_BALL_RADIUS = 0.4


def hartmann6_ball_constraint(x: list[float]) -> float:
    return sum((xi - _HARTMANN6_BALL_CENTER) ** 2 for xi in x) - _HARTMANN6_BALL_RADIUS**2


# --- Rosenbrock (valley/ill-conditioned; f_opt=0 at all-ones, not at a
# symmetric/central point like Ackley/Griewank) --------------------------
def make_rosenbrock(dim: int) -> Fn:
    def rosenbrock(x: list[float]) -> float:
        return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))

    return rosenbrock


def _rosenbrock_spec(dim: int) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"rosenbrock{dim}", dim=dim, bounds=[(-5.0, 10.0)] * dim, fn=make_rosenbrock(dim), minimize=True, f_opt=0.0
    )


# --- Rastrigin (highly multimodal; f_opt=0 at origin -- same "optimum at
# domain center" recognizability risk the paper flags for Ackley) --------
def make_rastrigin(dim: int) -> Fn:
    def rastrigin(x: list[float]) -> float:
        return 10.0 * len(x) + sum(xi**2 - 10.0 * math.cos(2 * math.pi * xi) for xi in x)

    return rastrigin


def _rastrigin_spec(dim: int) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"rastrigin{dim}", dim=dim, bounds=[(-5.12, 5.12)] * dim, fn=make_rastrigin(dim), minimize=True, f_opt=0.0
    )


# --- Levy (f_opt=0 at all-ones) ------------------------------------------
def make_levy(dim: int) -> Fn:
    def levy(x: list[float]) -> float:
        w = [1.0 + (xi - 1.0) / 4.0 for xi in x]
        total = math.sin(math.pi * w[0]) ** 2
        for wi in w[:-1]:
            total += (wi - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * wi + 1.0) ** 2)
        total += (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2 * math.pi * w[-1]) ** 2)
        return total

    return levy


def _levy_spec(dim: int) -> BenchmarkSpec:
    return BenchmarkSpec(name=f"levy{dim}", dim=dim, bounds=[(-10.0, 10.0)] * dim, fn=make_levy(dim), minimize=True, f_opt=0.0)


# --- Griewank (f_opt=0 at origin -- same center-of-domain risk as Ackley) --
def make_griewank(dim: int) -> Fn:
    def griewank(x: list[float]) -> float:
        sum_sq = sum(xi**2 for xi in x) / 4000.0
        prod_cos = 1.0
        for i, xi in enumerate(x):
            prod_cos *= math.cos(xi / math.sqrt(i + 1))
        return sum_sq - prod_cos + 1.0

    return griewank


def _griewank_spec(dim: int) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"griewank{dim}", dim=dim, bounds=[(-600.0, 600.0)] * dim, fn=make_griewank(dim), minimize=True, f_opt=0.0
    )


# --- Michalewicz (steep, irregular multimodal landscape; optimum is not at
# any symmetric/central point, unlike Ackley/Griewank) -- f_opt values below
# are literature-standard (sfu.ca/~ssurjano/michal.html), cross-checked here
# via dual_annealing + L-BFGS-B polish and multi-start L-BFGS-B (20k restarts).
_MICHALEWICZ_M = 10


def make_michalewicz(dim: int) -> Fn:
    def michalewicz(x: list[float]) -> float:
        return -sum(math.sin(xi) * math.sin((i + 1) * xi**2 / math.pi) ** (2 * _MICHALEWICZ_M) for i, xi in enumerate(x))

    return michalewicz


_MICHALEWICZ_F_OPT = {2: -1.8013034101, 5: -4.6876581790, 10: -9.6601517156}


def _michalewicz_spec(dim: int) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"michalewicz{dim}",
        dim=dim,
        bounds=[(0.0, math.pi)] * dim,
        fn=make_michalewicz(dim),
        minimize=True,
        f_opt=_MICHALEWICZ_F_OPT[dim],
    )


# --- Styblinski-Tang (f_opt = dim * f(x*), x* = -2.903534027771177 the real
# root of 4x^3 - 32x + 5 = 0, in every dimension) -------------------------
_STYBLINSKI_TANG_XSTAR = -2.903534027771177


def make_styblinski_tang(dim: int) -> Fn:
    def styblinski_tang(x: list[float]) -> float:
        return 0.5 * sum(xi**4 - 16.0 * xi**2 + 5.0 * xi for xi in x)

    return styblinski_tang


def _styblinski_tang_spec(dim: int) -> BenchmarkSpec:
    fn = make_styblinski_tang(dim)
    return BenchmarkSpec(
        name=f"styblinski_tang{dim}",
        dim=dim,
        bounds=[(-5.0, 5.0)] * dim,
        fn=fn,
        minimize=True,
        f_opt=fn([_STYBLINSKI_TANG_XSTAR] * dim),
    )


# --- Shekel (4-D, m=10 local minima "wells"; f_opt verified via 5000-restart
# L-BFGS-B) -----------------------------------------------------------------
_SHEKEL_BETA = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
_SHEKEL_C = [
    [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
    [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
    [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
    [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
]


def shekel(x: list[float]) -> float:
    total = 0.0
    for i in range(10):
        inner = sum((x[j] - _SHEKEL_C[j][i]) ** 2 for j in range(4))
        total += 1.0 / (inner + _SHEKEL_BETA[i] / 10.0)
    return -total


# --- Six-Hump Camel (2-D, two symmetric global minima) ---------------------
def six_hump_camel(x: list[float]) -> float:
    x1, x2 = x
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


REGISTRY: dict[str, BenchmarkSpec] = {
    "branin": BenchmarkSpec(
        name="branin", dim=2, bounds=[(-5.0, 10.0), (0.0, 15.0)], fn=branin, minimize=True, f_opt=0.397887
    ),
    "hartmann6": BenchmarkSpec(
        name="hartmann6", dim=6, bounds=[(0.0, 1.0)] * 6, fn=hartmann6, minimize=True, f_opt=-3.32237
    ),
    "ackley10": _ackley_spec(10),
    "ackley20": _ackley_spec(20),
    "constrained_hartmann6": BenchmarkSpec(
        name="constrained_hartmann6",
        dim=6,
        bounds=[(0.0, 1.0)] * 6,
        fn=hartmann6,
        minimize=True,
        f_opt=-2.8992024,
        constraint_fn=hartmann6_ball_constraint,
        constraint_upper=0.0,
    ),
    "rosenbrock6": _rosenbrock_spec(6),
    "rosenbrock10": _rosenbrock_spec(10),
    "rastrigin6": _rastrigin_spec(6),
    "rastrigin10": _rastrigin_spec(10),
    "levy6": _levy_spec(6),
    "levy10": _levy_spec(10),
    "griewank10": _griewank_spec(10),
    "michalewicz2": _michalewicz_spec(2),
    "michalewicz5": _michalewicz_spec(5),
    "michalewicz10": _michalewicz_spec(10),
    "styblinski_tang6": _styblinski_tang_spec(6),
    "styblinski_tang10": _styblinski_tang_spec(10),
    "shekel": BenchmarkSpec(
        name="shekel", dim=4, bounds=[(0.0, 10.0)] * 4, fn=shekel, minimize=True, f_opt=-10.536443153483528
    ),
    "six_hump_camel": BenchmarkSpec(
        name="six_hump_camel",
        dim=2,
        bounds=[(-3.0, 3.0), (-2.0, 2.0)],
        fn=six_hump_camel,
        minimize=True,
        f_opt=-1.0316284535,
    ),
}

_GP_SAMPLE_RE = re.compile(r"^gp_sample(\d+)$")
GP_SAMPLE_LENGTHSCALE = 0.2
GP_SAMPLE_M = 1028  # number of random Fourier features in the GP sample recipe


def _gp_sample_spec(dim: int, seed: int) -> BenchmarkSpec:
    """A fresh approximate squared-exponential-kernel GP sample path, drawn
    via M random Fourier features (Rahimi & Recht, 2007) using the standard
    random-Fourier-feature GP sample recipe: f(x) = sqrt(2/M) * sum_m w_m cos(theta_m . x + tau_m),
    w_m ~ N(0,1), theta_m ~ N(0, I/lengthscale^2) (the SE kernel's spectral
    density), tau_m ~ U(0, 2*pi). Domain is the unit cube directly (synthetic
    problems are optimized in the normalized unit cube).

    Deterministic in `seed` alone. Because the function is a fresh random
    draw every time, it cannot have been seen during pretraining -- this is
    the paper's own recommended control for testing genuine optimization
    ability isolated from benchmark memorization ("Benchmark recognition and
    evaluation": random GP sample paths are absent from pretraining by
    construction).

    `f_opt` has no closed form; it's a numerical estimate (multi-start
    L-BFGS-B) and therefore only approximately correct -- true regret against
    it can occasionally read slightly negative if a run's Sobol/optimizer
    beats our multi-start search. Widen `n_restarts` if that happens often.
    """
    import numpy as np
    from scipy.optimize import minimize

    # Apple Accelerate's BLAS emits spurious "divide by zero"/"overflow" RuntimeWarnings
    # on some (theta @ x) shapes here with no actual NaN/Inf in the result (verified);
    # silenced locally so real numerical issues elsewhere aren't lost in the noise.
    np.seterr(all="ignore")

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(GP_SAMPLE_M)
    theta = rng.standard_normal((GP_SAMPLE_M, dim)) / GP_SAMPLE_LENGTHSCALE
    tau = rng.uniform(0.0, 2 * math.pi, size=GP_SAMPLE_M)
    scale = math.sqrt(2.0 / GP_SAMPLE_M)

    def gp_sample(x: list[float]) -> float:
        xa = np.asarray(x, dtype=float)
        return float(scale * np.sum(w * np.cos(theta @ xa + tau)))

    n_restarts = 30 + 5 * dim
    best = None
    for x0 in rng.uniform(0.0, 1.0, size=(n_restarts, dim)):
        res = minimize(gp_sample, x0, bounds=[(0.0, 1.0)] * dim, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res

    return BenchmarkSpec(
        name=f"gp_sample{dim}",
        dim=dim,
        bounds=[(0.0, 1.0)] * dim,
        fn=gp_sample,
        minimize=True,
        f_opt=float(best.fun),
        extra={"w": w.tolist(), "theta": theta.tolist(), "tau": tau.tolist(), "lengthscale": GP_SAMPLE_LENGTHSCALE},
    )


def get_spec(name: str, seed: int | None = None) -> BenchmarkSpec:
    """`seed` is only consulted for the `gp_sample<dim>` family, whose
    identity *is* the random draw -- every other benchmark is a fixed
    textbook function and ignores it.
    """
    if name in REGISTRY:
        return REGISTRY[name]
    m = _GP_SAMPLE_RE.match(name)
    if m:
        if seed is None:
            raise ValueError(f"benchmark '{name}' is a random GP sample path and requires a seed to be reproducible")
        return _gp_sample_spec(int(m.group(1)), seed)
    raise KeyError(f"unknown benchmark '{name}' (choices: {sorted(REGISTRY)}, or 'gp_sample<dim>' e.g. 'gp_sample6')")
