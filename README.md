# AlphaBO

Plug-in **agentic Bayesian optimization**. An LLM agent (Sara) owns the campaign.
A BoTorch CLI backend (lenz) owns the trial log, posterior, and acquisition.
Classical BO methods occupy slots the agent can turn on, inspect, or override.

This is an independent reimplementation of Sara and lenz from [Agentic Bayesian
Optimization through Surrogate-Augmented Autoresearch](https://arxiv.org/abs/2608.00316)
(Brunzema et al., 2026). Not an official Meta repository. The original paper
has no released code.

AlphaBO adds a plugin protocol so methods wrap as backend capabilities the agent
invokes, which the Meta paper left as future work:

| Slot | Core default | Plugins |
|---|---|---|
| Surrogate \(M\) | fixed Matérn | **CAKE** (Suwandi et al., NeurIPS 2025) |
| Region \(B\) | box / `set-bounds` | **TuRBO** (Eriksson et al., 2019) |
| Prior | natural language only | **πBO** (Hvarfner et al., 2022) via `set-belief` |
| Sampler | BoTorch acqf opt | **LLAMBO** sampling (Liu et al., 2024) |

Sara still has only `bash` and `read`. New capability is a `lenz <plugin> ...`
verb plus a short prompt note. Method state lives in `state.json` under
`plugins`, not on the live shelf. Reconfiguring never discards trials.

`--no-lenz` drops the backend: Sara proposes configs and calls `./oracle`
herself (the test of whether the LLM can be the optimizer).

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Dependencies include `torch`, `gpytorch`, `botorch`, and
LLM clients (`anthropic`, `openai`).

## `lenz` (no LLM required)

```bash
lenz create --state ./state.json \
  --space '{"x1":{"kind":"range","lower":-5,"upper":10},"x2":{"kind":"range","lower":0,"upper":15}}' \
  --objectives '{"y":"minimize"}' \
  --acqf noisy_logei

lenz suggest --state ./state.json
lenz submit --state ./state.json --config '{"x1":1.0,"x2":2.0}' --metrics '{"y":12.3}'
lenz incumbent --state ./state.json
```

See `examples/branin/run_manual.py` and `sara/prompts/LENZ_REF.md` for a full
scripted loop and command reference.

## CAKE (Context-Aware Kernel Evolution)

CAKE evolves a GP kernel population during BO. LLMs act as crossover and
mutation operators. Each objective metric and each constraint metric gets its
own population. **BAKER** ranks weighted kernel combinations against the
configured acquisition function (logEI, NEHVI/EHVI, constrained EI, or
probability-of-feasibility before any feasible point is seen). If populations
are not ready yet, lenz falls back to the best kernel per metric.

CAKE is **not** part of Sara's agent loop. It runs inside `lenz` on its own
schedule during `observe`/`submit`, using `--kernel-llm-*` settings independent
of Sara's provider/model. Any model supported by `llm.factory` works. Tradeoff
is API cost and latency only.

```bash
lenz create --state ./state.json \
  --space '{"x1":{"kind":"range","lower":-5,"upper":10},"x2":{"kind":"range","lower":0,"upper":15}}' \
  --objectives '{"y":"minimize"}' \
  --surrogate cake --budget 30 \
  --kernel-llm-provider openai-compatible --kernel-llm-model your-model-id

lenz kernel-population --state ./state.json
lenz evolve-kernels --state ./state.json --force
```

Use with Sara by pinning cake at `lenz create`, or with scripted baselines via
`run_blind_baseline.py --policy cake`. See `LENZ_REF.md` for
`set-surrogate`, `evolve-kernels`, and `kernel-population`.

## `sara` (the agent)

Required flags:

- `--context`: markdown problem description
- `--eval`: shell command that evaluates one config (`sara` appends the config JSON)
- `--budget`: target number of evaluations
- `--workdir`: sandbox for `state.json` and `trace.jsonl` (e.g. `./results/logs/my-run`)

```bash
sara run \
  --provider anthropic --model claude-opus-5 \
  --context examples/branin/context.md \
  --eval "python3 examples/branin/eval.py" \
  --budget 30 \
  --workdir ./results/logs/branin-1
```

### LLM providers

Load credentials from `.env` (copy `.env.example`) or pass `--api-key` /
`--base-url`.

| `--provider` | Notes |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `openai-compatible` | Any OpenAI-compatible API. Requires `--base-url`. |
| `ollama` | Local Ollama at `http://localhost:11434/v1` by default |

If `OPENAI_API_KEY` is already set in your shell, `python-dotenv` will not
override it from `.env`. Use a distinct env var or pass `--api-key` explicitly.

The agent gets two tools (`bash`, `read`) in `--workdir`, matching the paper's
`pi --tools read,bash` setup. Prompts live in `sara/prompts/SYSTEM.md` and
`sara/prompts/LENZ_REF.md`. Budget is self-regulated by the agent, with a
1.5× budget reminder and a hard step cap in `sara/agent.py`.

## Benchmark harness

`benchmarks/` builds anti-memorization sandboxes: renamed parameters, unit-cube
encoding, and a shifted optimum. Scoring uses a hidden answer key the agent never
sees. List available functions:

```bash
python3 -c "from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))"
```

Real mixed-type HPO (not a textbook function): `bolt_lora` is the BOLT LoRA
emulator of Qwen3-8B fine-tuning (7 parameters: float / int / categorical).
It needs `pip install -e '.[bolt]'` once so the first evaluation can download
the Hugging Face weights. Scoring uses gap to BOLT's reported empirical best,
not a closed-form optimum. Blind mode renames parameters but keeps native
types; torus-shifting is disabled.

**Blind run** (hidden identity, scores true regret afterward):

```bash
python3 -m benchmarks.run_blind_test \
  --benchmark hartmann6 --provider anthropic --model claude-opus-5 \
  --budget 30 --root ./results/logs/blind-1
```

Pass `--no-lenz` for a Sara-only run: the LLM proposes every point, lenz is
never created, and the sandbox cannot import or invoke it. That is the probe
for "can the LLM act as the optimizer itself."

**Revealed run** (real benchmark name and bounds; see disclosure levels below):

```bash
python3 -m benchmarks.run_noblind_test \
  --benchmark hartmann6 --provider anthropic --model claude-opus-5 \
  --budget 5 --root ./results/logs/noblind-hartmann6
```

Pass `--shift` to keep the same relocated optimum as the blind sandbox. **Vanilla
baseline** (no LLM): `python3 -m benchmarks.run_blind_baseline --benchmark
hartmann6 --budget 30 --root ./results/logs/vanilla --policy vanilla`

Pin the backend for a whole run with `--surrogate {fixed,cake}` and `--acqf`.
CAKE also needs `--kernel-llm-provider` and `--kernel-llm-model`.

## Compare scripts

Compare scripts overlay regret curves across condition folders under
`results/logs/`. Defaults: budget `100`, seed `42`. Provider and model from
`scripts/_compare_env.sh`. `run_bolt_lora_compare.sh` is the regime-2 sweep (mixed-type HPO, no textbook
optimum): vanilla vs cake without Sara, then revealed Sara+lenz,
Sara+lenz+cake, and Sara-only (no lenz). Completed and in-flight legs are
skipped, so a crash-and-rerun does not redo vanilla or cake. Shift and
one-shot disclosure do not apply. To add Sara-only and cake-only onto
already-finished groups of *other* benchmarks, use `run_complement_backends.sh`.

| Script | What varies | Output dir |
|---|---|---|
| `run_benchmark_compare.sh <benchmark>` | backend (vanilla, sara-lenz, sara-lenz-cake) | `results/logs/<benchmark>-compare` |
| `run_benchmark_noblind_compare.sh <benchmark> [--shift-only]` | same backends, identity revealed | `results/logs/<benchmark>-noblind-compare-3config` |
| `run_noblind_compare.sh <benchmark> [--config …]` | disclosure level (see below) | `results/logs/<benchmark>-noblind-compare` |
| `run_bolt_lora_compare.sh [budget] [seed]` | vanilla, cake, sara-lenz, sara-lenz-cake, sara-only; skips completed | `results/logs/bolt_lora-compare` |
| `run_bolt_lora_followup.sh [list\|1\|2\|3\|all]` | wave 1: generic/misleading context (seed 42); wave 2: domain seeds 7/13; wave 3: seeds 5/11 if still needed | `results/logs/bolt_lora-generic-compare`, `-misleading-compare`, `-seedN-compare` |
| `run_complement_backends.sh [list\|run]` | add `sara-only` and `cake` into existing groups; skip completed/running | same dirs as above |
| `run_unclaimed.sh [list\|run] [--rq5]` | leftover legs not finished, in-flight, or queued by those parents | same dirs; RQ5 is new disclosure groups |

The first two scripts hold disclosure fixed and ask which backend wins. The third
holds the backend fixed and walks through all three disclosure levels.

### Disclosure levels

Used by `run_noblind_compare.sh` and the Python runners (`--shift`, revealed
`context.md`):

| Level | Identity | Optimum | Typical use |
|---|---|---|---|
| **Blind** | Hidden (renamed params, unit cube) | Shifted | Agentic BO with nothing to recall |
| **Revealed + shifted** | Real name and bounds | Same shift as blind | Does recalled structure help search? |
| **Revealed, unshifted** | Real name and bounds | Textbook location | One-shot recall probe |

Unshifted revealed runs are one-shot recall probes: use `run_noblind_test` with
default `--budget 5` and read `one_shot_success` at eval 1. Compare scripts use
budget 100 so blind, shifted, and unshifted curves share the same x-axis.

```bash
./scripts/run_benchmark_compare.sh hartmann6
./scripts/run_benchmark_noblind_compare.sh hartmann6
./scripts/run_benchmark_noblind_compare.sh hartmann6 100 42 --shift-only
./scripts/run_noblind_compare.sh hartmann6
./scripts/run_noblind_compare.sh hartmann6 100 42 --config sara-lenz-cake
./scripts/run_bolt_lora_compare.sh
./scripts/run_bolt_lora_compare.sh --list
./scripts/run_bolt_lora_compare.sh 100 42 --no-agent-only
./scripts/run_bolt_lora_compare.sh 100 42 --agent-only
./scripts/run_bolt_lora_followup.sh list
./scripts/run_bolt_lora_followup.sh 1
./scripts/run_bolt_lora_followup.sh 2
```

Override provider/model per run:

```bash
PROVIDER=anthropic MODEL=claude-opus-5 ./scripts/run_benchmark_compare.sh hartmann6
```

Other env vars: `ACQF`, `BASE_URL`, `API_KEY`, `EXTRA_BODY`, `KERNEL_LLM_*`,
`ONE_SHOT_TOL`. Plot all groups:

```bash
python3 -m benchmarks.plot_all
python3 -m benchmarks.plot_compare --root ./results/logs/hartmann6-compare
```

## Run viewer (`sara-viz`)

Local web UI for runs under `results/logs/` (or any `--root`):

```bash
sara-viz
sara-viz --root ./results/logs --port 9000
```

Tabs: Overview (convergence chart), Config, Trials, Trace, Tool use, Kernel
population (CAKE). **Experiments** mode overlays condition-level regret charts
(same data as `compare.html`). Compare mode overlays individual runs. Stdlib
only, no build step.

Runs write `run_meta.json` (provider, model, budget, timestamps) beside
`state.json` and `trace.jsonl`.

## Plugins

`lenz plugins` lists installed modules. Slot verbs:

```bash
lenz set-surrogate --state ./state.json --surrogate cake
lenz set-region --state ./state.json --policy turbo
lenz set-belief --state ./state.json --prior '{"x":{"dist":"normal","mu":0.3,"sigma":0.1}}'
lenz set-sampler --state ./state.json --sampler llambo
```

Method-specific verbs: `lenz turbo status`, `lenz evolve-kernels --force`,
`lenz llambo sample --n 8`. Each plugin is also a no-agent `--policy` in
`run_blind_baseline.py` when that makes sense (`vanilla`, `cake`, `turbo`).

Adding a method: implement `LenzPlugin` in `lenz/plugins/`, register it in
`lenz/plugins/registry.py`, ship a `PROMPT.md` next to it. Do not add fields
to `Shelf`.

## Extending

| Goal | Where to edit |
|---|---|
| New benchmark function | `benchmarks/functions.py`, `benchmarks/sandbox.py` |
| New vanilla baseline policy | `benchmarks/run_blind_baseline.py` (`POLICIES`) |
| New backend method | `lenz/plugins/` (slot + hooks + CLI verbs). Do not grow `Shelf`. |
| New compare condition | Add a block in a `scripts/run_*_compare.sh` |

`benchmarks/plot_compare.py` auto-discovers scored subdirectories. No code
change needed to add a new condition folder.

## Development

```bash
pip install -e ".[dev]"
pip install -e ".[bolt]"   # optional: BOLT LoRA HPO emulator
pytest
pytest -m "not slow"
```

## References

1. Brunzema, T., Tiao, J., Le, T., De Angeli, G., Xuan, Y., Gligorijevic, V.
   *Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch.*
   Meta, 2026.
   [Paper](https://arxiv.org/abs/2608.00316)

2. Suwandi, R., Yin, J., Wang, Y., Li, Y., Chang, Y., Theodoridis, S.
   *Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with
   LLMs.* NeurIPS 2025.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)

3. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., Poloczek, M.
   *Scalable Global Optimization via Local Bayesian Optimization.* NeurIPS 2019.
   [Code](https://github.com/uber-research/TuRBO)

4. Hvarfner, C., Stoll, D., Souza, A., Lindauer, M., Hutter, F., Nardi, L.
   *πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian
   Optimization.* ICLR 2022.

5. Liu, T., Astorga, N., Seedat, N., van der Schaar, M.
   *Large Language Models to Enhance Bayesian Optimization.* ICLR 2024.
   [Code](https://github.com/tennisonliu/LLAMBO)
