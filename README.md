# agentic-bo

Implementation of **agentic Bayesian optimization** from [Agentic Bayesian
Optimization through Surrogate-Augmented Autoresearch](https://arxiv.org/abs/2608.00316)
(2026), with **CAKE** (Context-Aware Kernel Evolution) integrated as a
first-class surrogate module from [Adaptive Kernel Design for Bayesian
Optimization Is a Piece of CAKE with
LLMs](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)
(NeurIPS 2025).

Standard BO fixes its policy before the first evaluation and never changes it.
Agentic BO puts an LLM agent in charge of surrogate, acquisition, bounds, and
objectives while a BoTorch backend still fits models and optimizes acquisition.
CAKE targets a separate bottleneck: which GP kernel to use as data arrives,
via LLM-guided crossover and mutation plus BAKER ranking.

Three pieces:

- **`lenz`**: stateless BoTorch/GPyTorch CLI backend. Each command loads
  `state.json`, runs one operation, saves, and prints one JSON line.
  Reconfiguring the problem never discards prior trials. Default surrogate is
  fixed Matérn (`--surrogate fixed`).
- **`CAKE`**: adaptive GP kernel evolution with LLMs. A population of
  composite kernels maintained on a schedule inside `lenz`, with **separate**
  `--kernel-llm-*` calls (no shared conversation with Sara). Enabled via
  `--surrogate cake`. Code: `lenz/cake.py`.
- **`sara`**: LLM agent that drives `lenz` with `bash` and `read` only,
  sandboxed to a working directory, using system prompts from the Meta paper
  appendix. Sara frames the problem, calls `lenz create`, and loops
  suggest/evaluate/observe. She can pin `fixed` or `cake` for the whole run.

This is an **independent from-scratch reimplementation** of the agentic BO stack.
The Meta paper has no released reference code. This repo is **not** an official
Meta implementation. See [References](#references) for citations.

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

**Blind run** (hidden identity, scores true regret afterward):

```bash
python3 -m benchmarks.run_blind_test \
  --benchmark hartmann6 --provider anthropic --model claude-opus-5 \
  --budget 30 --root ./results/logs/blind-1
```

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

Three shell scripts overlay regret curves across condition folders under
`results/logs/`. Defaults: budget `100`, seed `42`. Provider and model from
`scripts/_compare_env.sh`.

| Script | What varies | Output dir |
|---|---|---|
| `run_benchmark_compare.sh <benchmark>` | backend (vanilla, sara-lenz, sara-lenz-cake) | `results/logs/<benchmark>-compare` |
| `run_benchmark_noblind_compare.sh <benchmark> [--shift-only]` | same backends, identity revealed | `results/logs/<benchmark>-noblind-compare-3config` |
| `run_noblind_compare.sh <benchmark> [--config …]` | disclosure level (see below) | `results/logs/<benchmark>-noblind-compare` |

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

## Extending

| Goal | Where to edit |
|---|---|
| New benchmark function | `benchmarks/functions.py`, `benchmarks/sandbox.py` |
| New vanilla baseline policy | `benchmarks/run_blind_baseline.py` (`POLICIES`) |
| New lenz capability (acquisition, surrogate) | `lenz/acquisition.py`, `lenz/cake.py`, `lenz/models.py` |
| New compare condition | Add a block in a `scripts/run_*_compare.sh` |

`benchmarks/plot_compare.py` auto-discovers scored subdirectories. No code
change needed to add a new condition folder.

## Development

```bash
pip install -e ".[dev]"
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
