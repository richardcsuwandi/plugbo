import json
from pathlib import Path

import pytest

from benchmarks.priors import get_prior_fixture
from benchmarks.run_blind_baseline import run_blind_baseline


def test_bolt_prior_fixtures_are_isolated_and_context_specific():
    domain = get_prior_fixture("bolt_lora", "bolt-domain")
    misleading = get_prior_fixture("bolt_lora", "misleading")
    domain["lr"]["a"] = 99

    assert get_prior_fixture("bolt_lora", "domain")["lr"]["a"] == 2.0
    assert misleading["lora_target"]["probs"]["0"] == 0.85
    with pytest.raises(ValueError, match="only for bolt_lora"):
        get_prior_fixture("hartmann6", "domain")


def test_baseline_records_composed_slots(tmp_path):
    result = run_blind_baseline(
        benchmark_name="branin",
        budget=3,
        root=tmp_path,
        policy_name="vanilla",
        seed=42,
        warmup=2,
        reveal=True,
        region="turbo",
        prior={"x1": {"dist": "normal", "mu": 0.5, "sigma": 0.2}},
        decay_beta=8.0,
    )

    sandbox = Path(result["sandbox"])
    meta = json.loads((sandbox / "run_meta.json").read_text())
    state = json.loads((sandbox / "state.json").read_text())
    assert meta["surrogate"] == "fixed"
    assert meta["region"] == "turbo"
    assert meta["prior"] == "pibo"
    assert meta["prior_decay_beta"] == 8.0
    assert state["shelf"]["region"] == "turbo"
    assert state["shelf"]["prior"] == "pibo"
