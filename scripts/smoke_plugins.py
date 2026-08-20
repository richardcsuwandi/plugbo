#!/usr/bin/env python3
"""Smoke-test PlugBO plugin *capabilities* (no Sara).

Default (no network): registry, CLI verbs, TuRBO trust region, πBO scoring,
CAKE/LLAMBO wiring, and short Branin loops for vanilla / turbo / pibo.

  python3 scripts/smoke_plugins.py
  python3 scripts/smoke_plugins.py --live     # one CAKE evolve + one LLAMBO sample

`--live` uses PROVIDER/MODEL/BASE_URL/API_KEY from the environment (the
shell wrapper sources scripts/_compare_env.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lenz_loop import evaluate, lenz  # noqa: E402
from lenz.plugins.registry import all_plugins  # noqa: E402

BRANIN_SPACE = {
    "x1": {"kind": "range", "lower": -5.0, "upper": 10.0},
    "x2": {"kind": "range", "lower": 0.0, "upper": 15.0},
}
BRANIN_OBJECTIVES = {"y": "minimize"}
BRANIN_OPT = {"x1": 3.14159265, "x2": 2.275}
ORACLE = ROOT / "examples" / "branin" / "eval.py"

FAILED: list[str] = []


def _ok(name: str) -> None:
    print(f"  PASS  {name}")


def _fail(name: str, err: BaseException) -> None:
    FAILED.append(name)
    print(f"  FAIL  {name}: {err}")


def check(name: str, fn) -> None:
    try:
        fn()
        _ok(name)
    except Exception as err:
        _fail(name, err)


def create(state: Path, *extra: str) -> dict:
    args = [
        "--space",
        json.dumps(BRANIN_SPACE),
        "--objectives",
        json.dumps(BRANIN_OBJECTIVES),
        "--acqf",
        "noisy_logei",
        "--force",
        *extra,
    ]
    return lenz(state, "create", *args)


def step(state: Path) -> dict:
    config = lenz(state, "suggest")[0]["config"]
    metrics = evaluate(ORACLE, config)
    lenz(state, "submit", "--config", json.dumps(config), "--metrics", json.dumps(metrics))
    return {"config": config, "metrics": metrics}


def warmup(state: Path, n: int) -> None:
    for _ in range(n):
        step(state)


def llm_flags_from_env() -> list[str]:
    provider = os.environ.get("PROVIDER")
    model = os.environ.get("MODEL")
    if not provider or not model:
        raise RuntimeError("PROVIDER and MODEL must be set for --live (source scripts/_compare_env.sh)")
    flags = ["--llm-provider", provider, "--llm-model", model]
    base_url = os.environ.get("BASE_URL") or ""
    if base_url:
        flags += ["--llm-base-url", base_url]
    extra = os.environ.get("EXTRA_BODY") or ""
    if extra:
        flags += ["--llm-extra-body", extra]
    key_env = os.environ.get("KERNEL_LLM_API_KEY_ENV") or ""
    if key_env:
        flags += ["--llm-api-key-env", key_env]
    return flags


# -- checks -----------------------------------------------------------------


def check_registry() -> None:
    by_name = {p.name: p.slot for p in all_plugins()}
    assert by_name["cake"] == "surrogate"
    assert by_name["turbo"] == "region"
    assert by_name["pibo"] == "prior"
    assert by_name["llambo"] == "sampler"


def check_plugins_cli(state: Path) -> None:
    create(state)
    payload = lenz(state, "plugins")
    names = {p["name"] for p in payload["plugins"]}
    assert names >= {"cake", "turbo", "pibo", "llambo"}
    slots = payload["slots"]
    assert slots["surrogate"] == "fixed"
    assert slots["region"] == "box"
    assert slots["sampler"] == "botorch"
    assert slots["prior"] == "none"


def check_turbo(state: Path) -> None:
    create(state)
    warmup(state, 4)
    init = lenz(state, "turbo", "init")
    assert init["center"] is not None
    assert init["length"] > 0
    status = lenz(state, "status")
    assert status["region"] == "turbo"
    cand = lenz(state, "suggest")[0]
    assert "config" in cand
    before = lenz(state, "turbo", "status")["length"]
    for _ in range(4):
        step(state)
    after = lenz(state, "turbo", "status")
    assert after["center"] is not None
    # counters or length must have moved; a stuck TR means on_observe is dead
    moved = (
        after["length"] != before
        or after["success_count"]
        or after["failure_count"]
        or after["restarts"]
    )
    assert moved, f"TuRBO state did not update after observes: {after}"


def check_pibo(state: Path) -> None:
    create(state)
    warmup(state, 5)
    belief = {
        "x1": {"dist": "normal", "mu": BRANIN_OPT["x1"], "sigma": 0.4},
        "x2": {"dist": "normal", "mu": BRANIN_OPT["x2"], "sigma": 0.4},
    }
    result = lenz(state, "set-belief", "--prior", json.dumps(belief), "--decay-beta", "20")
    assert result.get("belief", belief)
    status = lenz(state, "status")
    assert status["prior"] == "pibo"
    ranked = lenz(
        state,
        "score",
        "--configs",
        json.dumps([BRANIN_OPT, {"x1": -5.0, "x2": 15.0}]),
        "--acqf",
        "noisy_logei",
    )
    near = ranked[0]["noisy_logei"]
    far = ranked[1]["noisy_logei"]
    assert near > far, f"πBO should prefer the prior mode: near={near} far={far}"


def check_cake_wiring(state: Path) -> None:
    create(state, "--surrogate", "cake", "--budget", "20")
    pop = lenz(state, "kernel-population")
    assert pop["targets"] == ["y"]
    warmup(state, 3)
    evolved = lenz(state, "evolve-kernels")
    assert evolved["evolved"] is False
    create(
        state,
        "--surrogate",
        "cake",
        "--budget",
        "20",
        "--llm-provider",
        "openai",
        "--llm-model",
        "smoke-placeholder",
    )
    status = lenz(state, "status")
    assert status["default_llm"]["model"] == "smoke-placeholder"
    cake = status.get("cake") or {}
    assert cake.get("kernel_llm_source") == "default"


def check_llambo_wiring(state: Path) -> None:
    create(state, "--llm-provider", "openai", "--llm-model", "smoke-placeholder")
    info = lenz(state, "llambo", "status")
    assert info["llm_source"] == "default"
    assert info["llm"]["model"] == "smoke-placeholder"
    lenz(state, "set-sampler", "--sampler", "llambo")
    assert lenz(state, "status")["sampler"] == "llambo"
    lenz(state, "set-sampler", "--sampler", "botorch")


def check_loop(state: Path, name: str, setup) -> None:
    create(state)
    warmup(state, 3)
    setup(state)
    for _ in range(5):
        step(state)
    inc = lenz(state, "incumbent")
    assert inc["metrics"] is not None, f"{name}: no incumbent after the loop"
    assert inc["metrics"]["y"] < 50.0, f"{name}: incumbent y={inc['metrics']['y']} looks broken"


def check_cake_live(state: Path) -> None:
    flags = llm_flags_from_env()
    create(state, "--surrogate", "cake", "--budget", "20", *flags)
    warmup(state, 6)
    result = lenz(state, "evolve-kernels", "--force")
    pop = lenz(state, "kernel-population")
    members = pop["populations"].get("y") or []
    assert result["evolved"] is True or members, f"CAKE did not evolve: {result} pop={pop}"
    assert members, "CAKE population is still empty after evolve-kernels --force"
    print(f"         cake generation={result.get('kernel_generation')} n={len(members)}")


def check_llambo_live(state: Path) -> None:
    flags = llm_flags_from_env()
    create(state, *flags)
    packed = lenz(state, "llambo", "sample", "--n", "2")
    assert isinstance(packed, list) and packed, f"LLAMBO sample returned {packed!r}"
    for row in packed:
        assert set(row["config"]) == {"x1", "x2"}
        assert -5.0 <= row["config"]["x1"] <= 10.0
        assert 0.0 <= row["config"]["x2"] <= 15.0
    print(f"         llambo n={len(packed)} first={packed[0]['config']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call the configured LLM once for CAKE and LLAMBO")
    parser.add_argument("--workdir", default=None, help="keep state.json here instead of a temp dir")
    args = parser.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="plugbo-smoke-"))
    workdir.mkdir(parents=True, exist_ok=True)
    state = workdir / "state.json"
    print(f"workdir: {workdir}")

    print("\n[wiring]")
    check("plugin registry", check_registry)
    check("lenz plugins", lambda: check_plugins_cli(state))
    check("TuRBO region + on_observe", lambda: check_turbo(state))
    check("πBO set-belief + score", lambda: check_pibo(state))
    check("CAKE wiring (no LLM call)", lambda: check_cake_wiring(state))
    check("LLAMBO wiring (no LLM call)", lambda: check_llambo_wiring(state))

    print("\n[branin loops, 8 evals]")
    check("loop vanilla", lambda: check_loop(state, "vanilla", lambda _: None))
    check("loop turbo", lambda: check_loop(state, "turbo", lambda s: lenz(s, "turbo", "init")))
    check(
        "loop pibo",
        lambda: check_loop(
            state,
            "pibo",
            lambda s: lenz(
                s,
                "set-belief",
                "--prior",
                json.dumps(
                    {
                        "x1": {"dist": "normal", "mu": BRANIN_OPT["x1"], "sigma": 1.0},
                        "x2": {"dist": "normal", "mu": BRANIN_OPT["x2"], "sigma": 1.0},
                    }
                ),
            ),
        ),
    )

    if args.live:
        print("\n[live LLM]")
        check("CAKE evolve-kernels --force", lambda: check_cake_live(state))
        check("LLAMBO sample --n 2", lambda: check_llambo_live(state))
    else:
        print("\nskipping --live (CAKE evolve / LLAMBO sample). Re-run with --live to hit the API.")

    print()
    if FAILED:
        print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
        return 1
    print("all plugin smokes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
