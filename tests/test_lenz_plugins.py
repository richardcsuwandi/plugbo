"""Plugin protocol: slots, TuRBO region, πBO belief, CLI discovery."""

from __future__ import annotations

import json
import subprocess
import sys

from lenz.plugins.pibo import prior_density
from lenz.plugins.registry import all_plugins, occupant
from lenz.plugins.turbo import TurboPlugin
from lenz.space import Encoder, SearchSpace
from lenz.state import Frame, Objective, Shelf


def _frame(n: int = 6) -> tuple[Frame, Encoder]:
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    shelf = Shelf(objectives=[Objective(metric="y", minimize=True)])
    frame = Frame(space=space, shelf=shelf)
    for i in range(n):
        x = i / (n - 1)
        frame.submit({"x": x}, {"y": (x - 0.3) ** 2})
    return frame, Encoder(space)


def _run_cli(state_path, command, *args):
    out = subprocess.run(
        [sys.executable, "-m", "lenz.cli", command, "--state", str(state_path), *args],
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


def test_plugins_register_expected_slots():
    by_slot = {p.slot: [] for p in all_plugins()}
    for p in all_plugins():
        by_slot.setdefault(p.slot, []).append(p.name)
    assert "cake" in by_slot["surrogate"]
    assert "turbo" in by_slot["region"]
    assert "pibo" in by_slot["prior"]
    assert "llambo" in by_slot["sampler"]


def test_create_exposes_slots(tmp_path):
    state = tmp_path / "state.json"
    payload = _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
    )
    assert payload["ok"] is True
    assert payload["result"]["region"] == "box"
    assert payload["result"]["sampler"] == "botorch"
    assert payload["result"]["prior"] == "none"


def test_plugins_command_lists_modules(tmp_path):
    state = tmp_path / "state.json"
    _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
    )
    payload = _run_cli(state, "plugins")
    names = {p["name"] for p in payload["result"]["plugins"]}
    assert names >= {"cake", "turbo", "pibo", "llambo"}


def test_turbo_bounds_shrink_around_incumbent():
    frame, encoder = _frame()
    frame.shelf.region = "turbo"
    plugin = TurboPlugin()
    blob = plugin.ensure(frame, encoder)
    blob["length"] = 0.25
    blob["failure_tolerance"] = 2
    bounds = plugin.active_bounds(frame, encoder)
    assert bounds is not None
    width = float((bounds[1] - bounds[0]).max())
    domain = float((encoder.domain_bounds[1] - encoder.domain_bounds[0]).max())
    assert width < domain
    assert occupant(frame, "region") is not None


def test_turbo_length_halves_after_failures():
    frame, encoder = _frame()
    plugin = TurboPlugin()
    frame.shelf.region = "turbo"
    blob = plugin.ensure(frame, encoder)
    blob["failure_tolerance"] = 2
    blob["length"] = 0.8
    start = blob["length"]
    # Two non-improving observations (worse than incumbent near 0.3).
    for x in (0.95, 0.9):
        trial = frame.submit({"x": x}, {"y": 1.0})
        plugin.on_observe(frame, trial)
    assert blob["length"] == start / 2


def test_set_belief_weights_the_mode():
    frame, _ = _frame()
    frame.shelf.prior = "pibo"
    frame.plugins["pibo"] = {
        "belief": {"x": {"dist": "normal", "mu": 0.3, "sigma": 0.05}},
        "decay_beta": 10.0,
        "t0": 0,
        "prior_floor": 1e-12,
    }
    near = prior_density(frame, {"x": 0.3}, frame.plugins["pibo"]["belief"])
    far = prior_density(frame, {"x": 0.9}, frame.plugins["pibo"]["belief"])
    assert near > far * 10


def test_categorical_belief_accepts_json_string_keys():
    space = SearchSpace.from_json({"choice": {"kind": "choice", "values": [0, 1]}})
    frame = Frame(space=space, shelf=Shelf(objectives=[Objective(metric="y", minimize=True)]))
    belief = {"choice": {"dist": "categorical", "probs": {"0": 0.8, "1": 0.2}}}

    assert prior_density(frame, {"choice": 0}, belief) == 0.8
    assert prior_density(frame, {"choice": 1}, belief) == 0.2


def test_set_belief_cli(tmp_path):
    state = tmp_path / "state.json"
    _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
    )
    payload = _run_cli(
        state,
        "set-belief",
        "--prior",
        json.dumps({"x": {"dist": "normal", "mu": 0.2, "sigma": 0.1}}),
        "--decay-beta",
        "8",
    )
    assert payload["ok"] is True
    loaded = json.loads(state.read_text())
    assert loaded["shelf"]["prior"] == "pibo"
    assert loaded["plugins"]["pibo"]["belief"]["x"]["mu"] == 0.2


def test_legacy_cake_shelf_migrates_into_plugins(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "space": {"x": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"}},
                "shelf": {
                    "objectives": [{"metric": "y", "minimize": True}],
                    "constraints": [],
                    "acqf": "noisy_logei",
                    "acqf_params": {},
                    "bounds": {},
                    "surrogate": "cake",
                    "kernel_populations": {"y": [{"expression": "SE", "bic": 1.0, "generation": 1}]},
                    "kernel_evolution_states": {"y": {"generation": 1, "frozen": False}},
                },
                "trials": [],
                "events": [],
            }
        )
    )
    frame = Frame.load(str(path))
    assert "cake" in frame.plugins
    assert frame.plugins["cake"]["kernel_populations"]["y"][0]["expression"] == "SE"
    assert "kernel_populations" not in frame.to_json()["shelf"]


def test_pibo_wraps_acquisition_scores():
    from lenz.optimize import score

    frame, encoder = _frame()
    frame.shelf.prior = "pibo"
    frame.shelf.acqf = "noisy_logei"
    frame.plugins["pibo"] = {
        "belief": {"x": {"dist": "normal", "mu": 0.3, "sigma": 0.05}},
        "decay_beta": 20.0,
        "t0": 0,
        "prior_floor": 1e-12,
    }
    ranked = score(frame, encoder, [{"x": 0.3}, {"x": 0.9}], ["noisy_logei"])
    assert ranked[0]["noisy_logei"] > ranked[1]["noisy_logei"]


def test_create_stores_default_llm(tmp_path):
    state = tmp_path / "state.json"
    payload = _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
        "--llm-provider",
        "openai",
        "--llm-model",
        "gpt-4.1",
    )
    assert payload["ok"] is True
    loaded = json.loads(state.read_text())
    assert loaded["default_llm"]["provider"] == "openai"
    assert loaded["default_llm"]["model"] == "gpt-4.1"


def test_create_inherits_agent_llm_sidecar(tmp_path):
    from lenz.llm_config import write_sidecar

    state = tmp_path / "state.json"
    write_sidecar(tmp_path, {"provider": "anthropic", "model": "claude-opus-5"})
    payload = _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
    )
    assert payload["ok"] is True
    loaded = json.loads(state.read_text())
    assert loaded["default_llm"]["provider"] == "anthropic"
    assert loaded["default_llm"]["model"] == "claude-opus-5"


def test_set_llm_cli(tmp_path):
    state = tmp_path / "state.json"
    _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
    )
    payload = _run_cli(state, "set-llm", "--provider", "openai", "--model", "gpt-4.1")
    assert payload["ok"] is True
    loaded = json.loads(state.read_text())
    assert loaded["default_llm"]["model"] == "gpt-4.1"


def test_kernel_llm_override_not_mixed_with_default(tmp_path):
    state = tmp_path / "state.json"
    _run_cli(
        state,
        "create",
        "--space",
        json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}}),
        "--objectives",
        json.dumps({"y": "minimize"}),
        "--surrogate",
        "cake",
        "--llm-provider",
        "anthropic",
        "--llm-model",
        "claude-opus-5",
        "--kernel-llm-provider",
        "openai",
        "--kernel-llm-model",
        "gpt-4.1",
    )
    loaded = json.loads(state.read_text())
    assert loaded["default_llm"]["model"] == "claude-opus-5"
    assert loaded["plugins"]["cake"]["kernel_llm"]["model"] == "gpt-4.1"


def test_maybe_evolve_inherits_default_llm(monkeypatch):
    from lenz import cake
    from llm.base import ChatResponse, LLMClient

    class FakeClient(LLMClient):
        def __init__(self):
            super().__init__(model="fake")
            self.calls = 0

        def chat(self, messages, tools, system):
            self.calls += 1
            return ChatResponse(
                content="Kernel: SE\nAnalysis: x", tool_calls=[], stop_reason="end_turn"
            )

    frame, encoder = _frame(n=8)
    frame.shelf.surrogate = "cake"
    frame.default_llm = {"provider": "openai", "model": "inherited"}
    client = FakeClient()
    seen = []

    def fake_client(spec, timeout=None):
        seen.append(spec)
        return client

    monkeypatch.setattr("lenz.cake.client_from_spec", fake_client)
    assert cake.maybe_evolve(frame, encoder, force=True) is True
    assert seen[0]["model"] == "inherited"
    assert client.calls >= 1


def test_maybe_evolve_kernel_override_wins(monkeypatch):
    from lenz import cake
    from llm.base import ChatResponse, LLMClient

    class FakeClient(LLMClient):
        def __init__(self):
            super().__init__(model="fake")
            self.calls = 0

        def chat(self, messages, tools, system):
            self.calls += 1
            return ChatResponse(
                content="Kernel: SE\nAnalysis: x", tool_calls=[], stop_reason="end_turn"
            )

    frame, encoder = _frame(n=8)
    frame.shelf.surrogate = "cake"
    frame.default_llm = {"provider": "openai", "model": "inherited"}
    cake.state(frame)["kernel_llm"] = {"provider": "anthropic", "model": "override"}
    client = FakeClient()
    seen = []

    def fake_client(spec, timeout=None):
        seen.append(spec)
        return client

    monkeypatch.setattr("lenz.cake.client_from_spec", fake_client)
    assert cake.maybe_evolve(frame, encoder, force=True) is True
    assert seen[0]["model"] == "override"


def test_llambo_inherits_default_llm(monkeypatch):
    from lenz.plugins.llambo import LlamboPlugin
    from llm.base import ChatResponse

    frame, _ = _frame(n=3)
    frame.default_llm = {"provider": "openai", "model": "inherited"}
    plugin = LlamboPlugin()
    class Dummy:
        def chat(self, messages, tools, system):
            return ChatResponse(content='[{"x": 0.4}]', tool_calls=[], stop_reason="end_turn")

    seen = []

    def fake_client(spec, timeout=None):
        seen.append(spec)
        return Dummy()

    monkeypatch.setattr("lenz.plugins.llambo.client_from_spec", fake_client)
    packed = plugin.sample(frame, Encoder(frame.space), n=1)
    assert packed[0]["config"]["x"] == 0.4
    assert seen[0]["model"] == "inherited"
    assert plugin.summary(frame)["llm_source"] == "default"

