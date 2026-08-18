# agentic-bo

Implements **agentic Bayesian optimization** from ["Agentic Bayesian
Optimization through Surrogate-Augmented Autoresearch"](https://arxiv.org/abs/2608.00316)
(Brunzema, Tiao, Le, De Angeli, Xuan, Gligorijevic, Meta, 2026), with
**CAKE** (Context-Aware Kernel Evolution) integrated as a first-class surrogate
module, from
["Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with
LLMs"](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)
(Suwandi, Yin, Wang, Li, Chang, Theodoridis, NeurIPS 2025).

Standard BO fixes its policy (surrogate, acquisition, bounds, objectives) before
the first evaluation and never changes it. Agentic BO puts an LLM agent in charge
of that policy while a BoTorch backend still fits the surrogate and optimizes
acquisition. CAKE addresses a different bottleneck: which GP kernel to use as
data arrives, via LLM-guided crossover and mutation plus BAKER ranking.

Three pieces:

- **`lenz`**: stateless BoTorch/GPyTorch CLI backend. Each command loads
  `state.json`, runs one operation, saves, and prints one JSON line.
  Reconfiguring the problem never discards prior trials. Default surrogate is
  fixed Matérn (`--surrogate fixed`).
- **`CAKE`**: adaptive GP kernel evolution ([NeurIPS
  2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)).
  A population of composite kernels maintained on a schedule inside `lenz`, with
  **separate** `--kernel-llm-*` calls (no shared conversation with Sara).
  Enabled via `--surrogate cake`. Code: `lenz/cake.py`.
- **`sara`**: LLM agent that drives `lenz` with `bash` and `read` only,
  sandboxed to a working directory, using system prompts from the Meta paper
  appendix. Sara frames the problem, calls `lenz create`, and loops
  suggest/evaluate/observe. She can pin `fixed` or `cake` for the whole run.

This is an **independent from-scratch reimplementation** of the agentic BO stack.
The Meta paper has no released reference code. This repo is **not** an official
Meta implementation. CAKE is integrated per the NeurIPS paper above.

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
mutation operators. **BAKER** (BIC-Acquisition Kernel Ranking) picks which
kernel to acquire with, balancing BIC fit against the configured acquisition
function.

Paper: [Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with
LLMs](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)
(NeurIPS 2025).

CAKE is **not** part of Sara's agent loop. It runs inside `lenz` on its own
schedule during `observe`/`submit`, using `--kernel-llm-*` settings independent
of Sara's provider/model. Any model supported by `llm.factory` works. Tradeoff
is API cost and latency only.

Requirements: single-objective, unconstrained studies (`--surrogate cake`).

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

`benchmarks/` implements the paper's anti-memorization sandbox (renamed
parameters, unit cube, shifted optimum, hidden identity) plus no-blind variants
(identity revealed). Benchmark list:

```bash
python3 -c "from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))"
```

**Blind agent run** (scores true regret after the run):

```bash
python3 -m benchmarks.run_blind_test \
  --benchmark hartmann6 --provider anthropic --model claude-opus-5 \
  --budget 30 --root ./results/logs/blind-1
```

**No-blind run** (revealed identity, default budget 5 for one-shot probes):

```bash
python3 -m benchmarks.run_noblind_test \
  --benchmark hartmann6 --provider anthropic --model claude-opus-5 \
  --budget 5 --root ./results/logs/noblind-hartmann6
```

Add `--shift` for revealed identity with a relocated optimum. **Vanilla baseline**
(no LLM): `python3 -m benchmarks.run_blind_baseline --benchmark hartmann6
--budget 30 --root ./results/logs/vanilla --policy vanilla`

Pin backend for the whole run with `--surrogate {fixed,cake}` and `--acqf`.
CAKE also needs `--kernel-llm-provider` and `--kernel-llm-model`.

What each probe measures and how to read curves is in
[`docs/observations.md`](docs/observations.md).

## Compare scripts

Three shell scripts overlay regret curves across condition subdirectories. Default
budget `100`, seed `42`. Provider/model from `scripts/_compare_env.sh`.

| Script | Varies | Output dir |
|---|---|---|
| `run_benchmark_compare.sh <benchmark>` | backend (vanilla, sara-lenz, sara-lenz-cake) | `results/logs/<benchmark>-compare` |
| `run_benchmark_noblind_compare.sh <benchmark> [--shift-only]` | same backends, identity revealed | `results/logs/<benchmark>-noblind-compare-3config` |
| `run_noblind_compare.sh <benchmark> [--config …]` | disclosure (blind, shifted, unshifted) | `results/logs/<benchmark>-noblind-compare` |

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
| Regime-2 (prior-informed) benchmark | See `docs/observations.md` |

`benchmarks/plot_compare.py` auto-discovers scored subdirectories. No code
change needed to add a new condition folder.

## Development

```bash
pip install -e ".[dev]"
pytest
pytest -m "not slow"
```

## Limitations

- Bounded reals, integers, and categoricals only (paper scope).
- CAKE: single-objective, unconstrained only.
- Kernel evolution runs synchronously inside scheduled `submit`/`observe` calls
  (up to two LLM calls, 90s timeout each in `lenz/cake.py`).
- Constrained BO feasibility phase uses independent constraint probabilities.
- No fine-tuning of the agent. General-purpose LLM plus system prompt only.
