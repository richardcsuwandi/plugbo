"""Tests for lenz/cake.py: CAKE population, BIC
fitness, LLM-guided crossover/mutation, BAKER ranking) and its scheduling.
No network calls -- LLM calls go through a scripted fake client.
"""

import json
import subprocess
import sys
import time

import pytest

from lenz import cake
from lenz.space import Encoder, SearchSpace
from lenz.state import Frame, Objective, Shelf
from llm.base import ChatResponse, LLMClient


# -- hard-timeout mechanism -----------------------------------------------
# Regression coverage for a real bug found during live testing: a client-level HTTP
# timeout does not bound a *streaming* response's total wall-clock time (its read
# timeout resets on every chunk), so a slow trickle can run far past the configured
# timeout. `_call_with_hard_timeout` enforces a real deadline independent of that.


def test_call_with_hard_timeout_returns_fast_result():
    assert cake._call_with_hard_timeout(lambda: 42, timeout_seconds=1.0) == 42


def test_call_with_hard_timeout_propagates_exceptions():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        cake._call_with_hard_timeout(boom, timeout_seconds=1.0)


def test_call_with_hard_timeout_bounds_a_hanging_call():
    def hangs_forever():
        time.sleep(10.0)
        return "should never get here"

    t0 = time.time()
    with pytest.raises(cake.LLMCallTimeout):
        cake._call_with_hard_timeout(hangs_forever, timeout_seconds=0.2)
    elapsed = time.time() - t0
    assert elapsed < 1.0  # returned promptly despite the underlying call still running


class FakeClient(LLMClient):
    """Replays a fixed sequence of responses; raises after exhausting them."""

    def __init__(self, responses=None, raises: Exception | None = None):
        super().__init__(model="fake")
        self._responses = list(responses) if responses is not None else None
        self._raises = raises
        self.calls = 0

    def chat(self, messages, tools, system):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = "Kernel: SE + PER\nAnalysis: plausible periodic structure"
        return ChatResponse(content=content, tool_calls=[], stop_reason="end_turn")


def _toy_frame(n: int = 8, surrogate: str = "cake") -> tuple[Frame, Encoder]:
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    shelf = Shelf(objectives=[Objective(metric="y", minimize=True)], surrogate=surrogate)
    frame = Frame(space=space, shelf=shelf)
    for i in range(n):
        x = i / (n - 1) if n > 1 else 0.5
        frame.submit({"x": x}, {"y": (x - 0.3) ** 2})
    return frame, Encoder(space)


# -- scheduling ---------------------------------------------------------


def test_should_evolve_false_when_not_cake():
    frame, _ = _toy_frame(n=10, surrogate="fixed")
    assert cake.should_evolve(frame, "y") is False


def test_should_evolve_respects_init_after():
    frame, _ = _toy_frame(n=5)
    frame.shelf.kernel_init_after = 6
    assert cake.should_evolve(frame, "y") is False
    frame, _ = _toy_frame(n=6)
    frame.shelf.kernel_init_after = 6
    assert cake.should_evolve(frame, "y") is True


def test_should_evolve_respects_evolve_every():
    frame, _ = _toy_frame(n=9)
    frame.shelf.kernel_populations["y"] = [{"expression": "SE", "bic": 1.0, "generation": 1}]
    frame.shelf.kernel_evolution_states["y"] = {
        "generation": 1,
        "last_evolved_at_n_observed": 6,
        "frozen": False,
    }
    frame.shelf.kernel_evolve_every = 4
    assert cake.should_evolve(frame, "y") is False  # 9 - 6 = 3 < 4
    frame.shelf.kernel_evolve_every = 3
    assert cake.should_evolve(frame, "y") is True  # 9 - 6 = 3 >= 3


def test_should_evolve_freezes_past_budget_fraction():
    frame, _ = _toy_frame(n=10)
    frame.shelf.budget = 20
    frame.shelf.kernel_freeze_fraction = 0.5
    assert cake.should_evolve(frame, "y") is False
    assert frame.shelf.kernel_evolution_states["y"]["frozen"] is True
    # frozen sticks even if evolve_every would otherwise fire again
    frame.shelf.kernel_evolution_states["y"]["last_evolved_at_n_observed"] = 0
    assert cake.should_evolve(frame, "y") is False


# -- evolve_generation ---------------------------------------------------


def test_evolve_generation_seeds_population_and_updates_state():
    frame, encoder = _toy_frame(n=8)
    client = FakeClient()
    cake.evolve_generation(frame, encoder, client, "y")
    population = frame.shelf.kernel_populations["y"]

    assert len(population) <= frame.shelf.kernel_population_size
    assert all(m["bic"] is not None for m in population)
    state = frame.shelf.kernel_evolution_states["y"]
    assert state["generation"] == 1
    assert state["last_evolved_at_n_observed"] == 8
    assert cake.get_best_kernel(frame, "y") is not None
    assert client.calls >= 1


def test_evolve_generation_survives_llm_failure():
    frame, encoder = _toy_frame(n=8)
    client = FakeClient(raises=RuntimeError("network exploded"))
    cake.evolve_generation(frame, encoder, client, "y")  # must not raise

    population = frame.shelf.kernel_populations["y"]
    assert population
    assert all(isinstance(m["bic"], float) for m in population)
    assert frame.shelf.kernel_evolution_states["y"]["generation"] == 1


def test_evolve_generation_is_upsert_not_duplicate():
    frame, encoder = _toy_frame(n=8)
    client = FakeClient(responses=["Kernel: SE\nAnalysis: x"] * 4)
    cake.evolve_generation(frame, encoder, client, "y")
    expressions = [m["expression"] for m in frame.shelf.kernel_populations["y"]]
    assert len(expressions) == len(set(expressions))  # no duplicate expressions


# -- maybe_evolve ---------------------------------------------------------


def test_maybe_evolve_skips_when_llm_not_configured():
    frame, encoder = _toy_frame(n=8)
    assert frame.shelf.kernel_llm == {}
    ran = cake.maybe_evolve(frame, encoder)
    assert ran is False
    assert frame.shelf.kernel_populations == {}
    assert frame.events[-1]["status"] == "skipped"


def test_maybe_evolve_force_ignores_schedule():
    frame, _ = _toy_frame(n=1)  # nowhere near kernel_init_after
    encoder = Encoder(frame.space)
    frame.shelf.kernel_llm = {"provider": "does-not-matter"}  # missing "model" -> still skipped, but exercises force path
    ran = cake.maybe_evolve(frame, encoder, force=True)
    assert ran is False  # only 1 observation, need >= 2 to fit anything


# -- BAKER ranking ---------------------------------------------------------


def test_baker_suggest_raises_when_population_empty():
    frame, encoder = _toy_frame(n=8)
    with pytest.raises(cake.CakeNotReadyError):
        cake.baker_suggest(frame, encoder, q=1, X_pending=None)


def test_baker_suggest_and_score_after_evolution():
    frame, encoder = _toy_frame(n=8)
    cake.evolve_generation(frame, encoder, FakeClient(), "y")
    population = frame.shelf.kernel_populations["y"]

    suggestions = cake.baker_suggest(frame, encoder, q=2, X_pending=None)
    assert len(suggestions) <= 2
    for s in suggestions:
        assert 0.0 <= s["config"]["x"] <= 1.0
        assert s["kernel"] in [m["expression"] for m in population]
        assert "baker_score" in s["acquisition_values"]

    scores = cake.baker_score(frame, encoder, [{"x": 0.3}, {"x": 0.9}], ["noisy_logei"])
    assert len(scores) == 2
    assert all("noisy_logei" in s for s in scores)


def test_suggest_dispatches_to_baker_end_to_end():
    from lenz import optimize as opt

    frame, encoder = _toy_frame(n=8)
    cake.evolve_generation(frame, encoder, FakeClient(), "y")

    result = opt.suggest(frame, encoder, q=1)
    assert "kernel" in result[0]


def test_suggest_falls_back_when_cake_not_ready():
    from lenz import optimize as opt

    frame, encoder = _toy_frame(n=8)  # surrogate=cake but no evolution has run
    result = opt.suggest(frame, encoder, q=1)
    assert "kernel" not in result[0]  # fell back to the normal single-model path


# -- CLI: create validation ---------------------------------------------


def _run_cli(state_path, command, *args):
    out = subprocess.run(
        [sys.executable, "-m", "lenz.cli", command, "--state", str(state_path), *args],
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


def test_create_cake_accepts_constraints(tmp_path):
    state = tmp_path / "state.json"
    payload = _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
        "--constraints",
        json.dumps([{"metric": "c1", "upper": 0.5}]),
        "--surrogate",
        "cake",
        "--budget",
        "20",
    )
    assert payload["ok"] is True
    assert payload["result"]["surrogate"] == "cake"


def test_create_cake_accepts_multi_objective(tmp_path):
    state = tmp_path / "state.json"
    payload = _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y1": "minimize", "y2": "maximize"}),
        "--surrogate",
        "cake",
        "--acqf",
        "nehvi",
        "--budget",
        "20",
    )
    assert payload["ok"] is True


def test_evolve_generation_per_constraint_metric():
    from lenz.state import Constraint

    frame, encoder = _toy_frame(n=8)
    frame.shelf.constraints = [Constraint(metric="c1", upper=0.5)]
    for i, t in enumerate(frame.observed_trials()):
        t.metrics["c1"] = 0.1 + i * 0.05
    client = FakeClient()
    cake.evolve_generation(frame, encoder, client, "y")
    cake.evolve_generation(frame, encoder, client, "c1")
    assert "y" in frame.shelf.kernel_populations
    assert "c1" in frame.shelf.kernel_populations
    assert cake.get_best_kernel(frame, "c1") is not None


def test_baker_constrained_uses_objective_and_constraint_kernels():
    from lenz import optimize as opt
    from lenz.state import Constraint

    frame, encoder = _toy_frame(n=8)
    frame.shelf.constraints = [Constraint(metric="c1", upper=0.5)]
    for i, t in enumerate(frame.observed_trials()):
        t.metrics["c1"] = 0.1 + i * 0.02
    client = FakeClient()
    cake.evolve_generation(frame, encoder, client, "y")
    cake.evolve_generation(frame, encoder, client, "c1")

    result = opt.suggest(frame, encoder, q=1)
    assert "kernel" in result[0]
    assert "baker_score" in result[0]["acquisition_values"]


def test_kernel_combos_single_target_is_full_population():
    frame, encoder = _toy_frame(n=8)
    client = FakeClient()
    cake.evolve_generation(frame, encoder, client, "y")
    observed = frame.observed_trials()
    X = __import__("torch").stack([encoder.encode(t.config) for t in observed])
    prepared = cake._prepared_populations(frame, encoder, X, observed, ["y"])
    combos = cake._kernel_combos(prepared)
    assert len(combos) == len(frame.shelf.kernel_populations["y"])


def test_create_cake_ok_single_objective_unconstrained(tmp_path):
    state = tmp_path / "state.json"
    payload = _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
        "--surrogate",
        "cake",
        "--budget",
        "20",
    )
    assert payload["ok"] is True
    assert payload["result"]["surrogate"] == "cake"
