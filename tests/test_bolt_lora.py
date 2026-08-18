"""BOLT LoRA HPO harness: mixed-type space, sandbox context, and (slow) emulator."""

import json
import subprocess
import sys

import pytest

from benchmarks.functions import get_spec, true_regret
from benchmarks.obfuscate import ObfuscatedBenchmark, build_obfuscated
from benchmarks.sandbox import build_sandbox
from viz.captions import bench_label

IDENTIFYING = ["lora", "qwen", "bolt", "hpo", "chewwt"]


def test_bolt_lora_spec_is_mixed_maximize():
    spec = get_spec("bolt_lora")
    assert spec.dim == 7
    assert spec.minimize is False
    assert spec.allow_shift is False
    assert spec.space is not None
    assert list(spec.space) == [
        "lr",
        "batch",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_layers",
        "lora_target",
    ]
    assert spec.space["lora_target"]["kind"] == "choice"
    assert spec.space["batch"]["type"] == "int"


def test_true_regret_maximize_and_minimize():
    spec = get_spec("bolt_lora")
    assert true_regret(spec, spec.f_opt) == pytest.approx(0.0)
    assert true_regret(spec, spec.f_opt - 0.1) == pytest.approx(0.1)
    branin = get_spec("branin")
    assert true_regret(branin, branin.f_opt + 0.5) == pytest.approx(0.5)


def test_bolt_lora_blind_keeps_types_hides_identity():
    ob = build_obfuscated("bolt_lora", seed=7, reveal=False)
    assert ob.param_names != ["lr", "batch", "lora_rank", "lora_alpha", "lora_dropout", "lora_layers", "lora_target"]
    assert all(abs(s) < 1e-12 for s in ob.shift_frac)
    space = ob.unit_space_json()
    assert space[ob.param_names[0]]["kind"] == "range"
    assert space[ob.param_names[0]]["type"] == "float"
    assert space[ob.param_names[1]]["type"] == "int"
    target = space[ob.param_names[6]]
    assert target["kind"] == "choice"
    assert target["values"] != [0, 1, 2, 3]
    assert ob.objectives_json() == {"y": "maximize"}


def test_bolt_lora_reveal_uses_real_names_and_choice_ints():
    ob = build_obfuscated("bolt_lora", seed=7, reveal=True, shift=True)
    assert ob.param_names == [
        "lr",
        "batch",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_layers",
        "lora_target",
    ]
    space = ob.unit_space_json()
    assert space["lora_target"]["values"] == [0, 1, 2, 3]
    assert space["lora_layers"]["type"] == "int"
    # shift is a no-op even if requested
    assert all(abs(s) < 1e-12 for s in ob.shift_frac)
    cfg = ob.true_to_unit([0.31100, 2, 4, 2, 0.87056, 30, 1])
    assert cfg["lora_target"] == 1
    assert cfg["batch"] == 2
    assert ob.unit_to_true(cfg) == [0.31100, 2, 4, 2, 0.87056, 30, 1]


def test_bolt_lora_secret_roundtrip():
    ob = build_obfuscated("bolt_lora", seed=3)
    restored = ObfuscatedBenchmark.from_secret(ob.to_secret())
    assert restored.param_names == ob.param_names
    assert restored.choice_values == ob.choice_values
    assert restored.spec.name == "bolt_lora"


def test_sandbox_bolt_lora_blind_hides_identity(tmp_path):
    built = build_sandbox("bolt_lora", root=tmp_path, seed=11)
    sandbox = built["sandbox"]
    context = (sandbox / "context.md").read_text()
    dirname = sandbox.name
    for s in IDENTIFYING:
        assert s not in context.lower()
        assert s not in dirname.lower()
    assert "Maximize" in context
    assert "categorical" in context
    assert "integer" in context

    ob = ObfuscatedBenchmark.from_secret(json.loads(built["secret_path"].read_text()))
    assert ob.objectives_json() == {"y": "maximize"}
    space = ob.unit_space_json()
    assert all(name not in space for name in ["lr", "lora_rank", "lora_target"])


def test_sandbox_bolt_lora_reveal_names_the_task(tmp_path):
    built = build_sandbox("bolt_lora", root=tmp_path, seed=11, reveal=True)
    context = (built["sandbox"] / "context.md").read_text()
    assert "lora" in context.lower()
    assert "0.34647" not in context
    assert "0.31100" not in context
    assert "`lr`" in context
    assert "Maximize" in context


def test_sandbox_bolt_lora_generic_context_keeps_types_drops_story(tmp_path):
    built = build_sandbox("bolt_lora", root=tmp_path, seed=11, reveal=True, context_variant="generic")
    context = (built["sandbox"] / "context.md").read_text()
    assert "`lr`" in context
    assert "`lora_target`" in context
    assert "integer" in context
    assert "categorical" in context
    assert "qwen" not in context.lower()
    assert "attention" not in context.lower()
    assert "0.34647" not in context
    assert "0.31100" not in context


def test_sandbox_bolt_lora_misleading_context_has_false_priors(tmp_path):
    built = build_sandbox("bolt_lora", root=tmp_path, seed=11, reveal=True, context_variant="misleading")
    context = (built["sandbox"] / "context.md").read_text()
    assert "0.05" in context
    assert "`lora_target` = 0" in context
    assert "fewer" in context.lower()
    assert "0.34647" not in context
    assert "0.31100" not in context
    assert "0.87056" not in context


def test_bench_label_bolt_lora():
    assert bench_label("bolt_lora") == "BOLT LoRA HPO"


def test_existing_continuous_blind_space_unchanged():
    ob = build_obfuscated("branin", seed=1)
    for d in ob.unit_space_json().values():
        assert d == {"kind": "range", "lower": 0.0, "upper": 1.0}


def test_hf_download_retries_ssl_then_succeeds(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from benchmarks import bolt_lora as m

    calls = {"n": 0}

    def fake_download(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("SSL: UNEXPECTED_EOF_WHILE_READING")
        return "/tmp/model.safetensors"

    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    path = m._hf_download("model.safetensors")
    assert path == "/tmp/model.safetensors"
    assert calls["n"] == 2


def test_hf_download_falls_back_to_mirror(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from benchmarks import bolt_lora as m

    endpoints: list[str | None] = []

    def fake_download(**kwargs):
        endpoints.append(kwargs.get("endpoint"))
        if kwargs.get("endpoint") is None:
            raise OSError("Max retries exceeded with url: huggingface.co SSL EOF")
        return "/tmp/from-mirror.safetensors"

    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("BOLT_HF_ENDPOINT", raising=False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    path = m._hf_download("config.json")
    assert path == "/tmp/from-mirror.safetensors"
    assert None in endpoints
    assert m.HF_MIRROR in endpoints


@pytest.mark.slow
def test_bolt_lora_emulator_and_oracle_match(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    spec = get_spec("bolt_lora")
    y = spec.fn([0.31100, 2, 4, 2, 0.87056, 30, 1])
    assert y == pytest.approx(spec.f_opt, abs=5e-3)

    built = build_sandbox("bolt_lora", root=tmp_path, seed=5, reveal=True)
    sandbox = built["sandbox"]
    ob = ObfuscatedBenchmark.from_secret(json.loads(built["secret_path"].read_text()))
    config = ob.true_to_unit([0.5, 3, 3, 3, 0.2, 10, 2])
    expected = ob.evaluate(config)
    out = subprocess.run(
        [sys.executable, str(sandbox / "oracle"), json.dumps(config)],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)["y"]
    assert got == pytest.approx(expected, rel=1e-6)
