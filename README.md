# PlugBO

A modular framework for **agentic Bayesian optimization**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/richardcsuwandi/plugbo/actions/workflows/tests.yml/badge.svg)](https://github.com/richardcsuwandi/plugbo/actions/workflows/tests.yml)

[Install](#install) · [Plugins](#plugins) · [Experiments](#experiments) · [Contributing](#contributing)

![PlugBO architecture](assets/architecture.svg)

Bayesian optimization (BO) uses a probabilistic surrogate to choose each
evaluation, but its overall search strategy is usually fixed in advance.
Agentic BO places an LLM agent at the center of the loop: it reviews trial data,
surrogate diagnostics, and natural-language context, then decides how to search
next. As evidence accumulates, the agent can query or override backend
suggestions and revise the surrogate, acquisition function, search bounds,
objectives, or constraints [[1]](#ref-1).

Meta's *Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch*
instantiates that idea as Sara, a campaign LLM, calling lenz, a BoTorch CLI that
owns the trial log, posterior, and acquisition [[1]](#ref-1) [[6]](#ref-6).
PlugBO keeps that agent and CLI, then turns the backend into a plugin surface:
surrogate, region, prior, and sampler are slots. Classical BO methods wrap as
`lenz` verbs the agent can enable, inspect, or override. The arrangement is
analogous to [MCP](https://modelcontextprotocol.io/): Sara is the host, with
only `bash` and `read`, and lenz is the shared tool surface. A BO method
registers extra verbs there the way an MCP server registers tools, so the agent
stays fixed while the surrogate, region, prior, or sampler can be swapped in as
a plugin.

> Note: This is an independent re-implementation of Sara and lenz [[1]](#ref-1).
> It is not an official Meta repository.

#### Why PlugBO?

* Sara stays a two-tool agent (`bash`, `read`). New capability is a `lenz`
  verb, not a new agent tool, so the agent surface does not grow per method.
* Classical BO methods (CAKE, TuRBO, πBO, LLAMBO) occupy plugin slots
  (surrogate, region, prior, sampler) instead of forking the optimization
  loop.
* Ships an anti-memorization benchmarking harness: blind/revealed disclosure
  levels and `gp_sample<dim>` no-prior controls, so an LLM-in-the-loop
  comparison is not secretly a retrieval test.
* Cross-seed aggregation and a merged viewer (`viz/aggregate.py`,
  `viz/merged_server.py`) read a condition as a mean ± SE band across seeds
  instead of one noisy line.

---

## Install

Not on PyPI. Two ways to install from source (Python 3.10+):

**Editable install**

```bash
git clone https://github.com/richardcsuwandi/plugbo.git
cd plugbo
pip install -e .
```

That pulls `torch`, `gpytorch`, `botorch`, and LLM clients (`anthropic`,
`openai`). Copy `.env.example` to `.env` if you will call an LLM.

**Dev install** (tests, plus the LoRA HPO emulator extra)

```bash
pip install -e ".[dev]"    # tests
pip install -e ".[bolt]"   # LoRA HPO emulator (BoLT)
```

---

## Plugins

Sara, lenz, and the BO methods are the same kind of piece: modules on one
control plane. Sara is the host (`bash` and `read`). lenz is the tool surface.
CAKE, TuRBO, πBO, and LLAMBO occupy slots on that surface.

| Module | Role | Default | Occupant | Commands |
|---|---|---|---|---|
| `sara` | campaign agent | — | — | `sara run` |
| `lenz` | trial log, posterior, acquisition | BoTorch loop | — | `create`, `suggest`, `submit`, `incumbent` |
| Surrogate | GP | fixed Matérn | **CAKE** [[2]](#ref-2) | `set-surrogate`, `evolve-kernels`, `kernel-population` |
| Region | search bounds | box | **TuRBO** [[3]](#ref-3) | `set-region`, `set-bounds`, `turbo status` |
| Prior | belief | none | **πBO** [[4]](#ref-4) | `set-belief` |
| Sampler | candidates | BoTorch | **LLAMBO** [[5]](#ref-5) | `set-sampler`, `llambo sample` |

Occupying a slot is a `set-*` verb. Plugins may add more verbs. Method state
lives in `state.json` under `plugins`, not on the live shelf. Reconfiguring
never discards trials. `lenz plugins` lists installed modules.

Vanilla loop (no LLM):

```bash
lenz create --state ./state.json \
  --space '{"x1":{"kind":"range","lower":-5,"upper":10},"x2":{"kind":"range","lower":0,"upper":15}}' \
  --objectives '{"y":"minimize"}' \
  --acqf noisy_logei

lenz suggest --state ./state.json
lenz submit --state ./state.json --config '{"x1":1.0,"x2":2.0}' --metrics '{"y":12.3}'
lenz incumbent --state ./state.json
```

Agent in the loop:

```bash
sara run \
  --provider anthropic --model claude-opus-5 \
  --context examples/branin/context.md \
  --eval "python3 examples/branin/eval.py" \
  --budget 30 \
  --workdir ./results/logs/branin-1
```

`--context` is problem markdown, `--eval` is a command that takes one config
JSON. `--no-lenz` drops the backend: Sara proposes configs and calls
`./oracle` herself.

| `--provider` | Notes |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `openai-compatible` | Requires `--base-url` |
| `ollama` | `http://localhost:11434/v1` by default |

Occupy a slot, with or without the agent:

```bash
lenz set-surrogate --state ./state.json --surrogate cake
lenz set-region --state ./state.json --policy turbo
lenz set-belief --state ./state.json --prior '{"x":{"dist":"normal","mu":0.3,"sigma":0.1}}'
lenz set-sampler --state ./state.json --sampler llambo
```

```bash
lenz create --state ./state.json \
  --space '{"x1":{"kind":"range","lower":-5,"upper":10},"x2":{"kind":"range","lower":0,"upper":15}}' \
  --objectives '{"y":"minimize"}' \
  --surrogate cake --budget 30
```

CAKE and LLAMBO inherit Sara's provider and model. Pass `--kernel-llm-*` or
`--sampler-llm-*` only to use a different model. TuRBO and πBO do not call an
LLM. CAKE evolves a GP kernel population inside `lenz` during
`observe`/`submit` [[2]](#ref-2); that trace is not part of Sara's conversation.

Command reference: `sara/prompts/LENZ_REF.md`. To add a method, implement
`LenzPlugin` in `lenz/plugins/`, ship a `{name}.md` prompt beside it, and
register it in `registry.py`. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Experiments

Synthetic functions use an anti-memorization sandbox (renamed parameters, unit
cube, shifted optimum). Scoring uses a hidden answer key. List functions:

```bash
python3 -c "from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))"
```

```bash
./scripts/run_synthetic.sh hartmann6
./scripts/run_synthetic.sh hartmann6 --disclosure revealed
./scripts/run_synthetic.sh hartmann6 --disclosure revealed-shift
./scripts/run_synthetic.sh hartmann6 --backend sara-lenz --disclosure all
./scripts/run_synthetic.sh hartmann6 --warmup 7
```

`--disclosure` is `blind` (default), `revealed`, `revealed-shift`, or `all`.
`--backend` is a comma list (`vanilla`, `cake`, `turbo`, `sara-lenz`,
`sara-lenz-cake`, `sara-only`), or `all` for vanilla + sara-lenz +
`sara-lenz-cake`. `--disclosure all` needs a single backend. Every backend
uses the same seeded Sobol warm-start, defaulting to `d+1` evaluations when a
seed is set. Pass `--warmup N` to override it for all selected backends.

To rerun only Sara-only in an existing synthetic comparison with the same
warm-start:

```bash
./scripts/rerun_sara_only.sh hartmann6
```

### LoRA optimization

The mixed-type experiment is LoRA hyperparameter optimization on
BoLT (Black-box Optimization for LLM Tasks) [[7]](#ref-7), a benchmark
of expensive LLM objectives (always revealed, no textbook optimum):

```bash
pip install -e '.[bolt]'
./scripts/run_bolt.sh
./scripts/run_bolt.sh --backend vanilla,cake
./scripts/run_bolt.sh --context generic
```

`--context` is `domain` (default), `generic`, or `misleading`.

Provider and model come from `.env` or `PROVIDER` / `MODEL`. Completed and
in-flight legs are skipped. Plots land in `compare.html` next to the run, or
open `sara-viz`.

---

## Run viewer (`sara-viz`)

```bash
sara-viz
sara-viz --root ./results/logs --port 9000
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"
./scripts/smoke_plugins.sh
```

Optional smoke flags and what they cover are in
[CONTRIBUTING.md](CONTRIBUTING.md). The script header is the source of truth
for `--live`, `--baseline`, and any later flags.

---

## Contributing

Bug reports, new BO plugins, and benchmarks are welcome.

- [CONTRIBUTING.md](CONTRIBUTING.md): setup, plugin protocol, adding a function
- [Issues](https://github.com/richardcsuwandi/plugbo/issues): bugs and proposals
- [Code of Conduct](CODE_OF_CONDUCT.md)

Please open an issue before a new slot, a large dependency, or a new
experiment backend.

---

## Citation

If you use PlugBO, please cite this repository (GitHub's cite button uses
[`CITATION.cff`](CITATION.cff)) and the papers for Sara/lenz and any plugins
you enable. The Sara/lenz design is Brunzema et al. [[1]](#ref-1). This repo
is an independent re-implementation.

---

## License

MIT. See [LICENSE](LICENSE). Contributions are under the same license.

---

## References

1. <span id="ref-1"></span> Brunzema, P., Tiao, L., Le, N., De Angeli, K., Xuan, Y.,
   Gligorijevic, D.
   *Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch.*
   arXiv:2608.00316, 2026.
   [Paper](https://arxiv.org/abs/2608.00316)
   No official code (this repo is an independent re-implementation).

2. <span id="ref-2"></span> Suwandi, R. C., Yin, F., Wang, J., Li, R., Chang, T.-H.,
   Theodoridis, S.
   *Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with
   LLMs.* NeurIPS 2025.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)
   [Code](https://github.com/richardcsuwandi/cake)

3. <span id="ref-3"></span> Eriksson, D., Pearce, M., Gardner, J., Turner, R. D.,
   Poloczek, M.
   *Scalable Global Optimization via Local Bayesian Optimization.* NeurIPS 2019.
   [Paper](https://arxiv.org/abs/1910.01739)
   [Code](https://github.com/uber-research/TuRBO)

4. <span id="ref-4"></span> Hvarfner, C., Stoll, D., Souza, A., Lindauer, M.,
   Hutter, F., Nardi, L.
   *πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian
   Optimization.* ICLR 2022.
   [Paper](https://arxiv.org/abs/2204.11051)
   [Code](https://github.com/piboauthors/PiBO-Spearmint)

5. <span id="ref-5"></span> Liu, T., Astorga, N., Seedat, N., van der Schaar, M.
   *Large Language Models to Enhance Bayesian Optimization.* ICLR 2024.
   [Paper](https://arxiv.org/abs/2402.03921)
   [Code](https://github.com/tennisonliu/LLAMBO)

6. <span id="ref-6"></span> Balandat, M., Karrer, B., Jiang, D. R., Daulton, S.,
   Letham, B., Wilson, A. G., Bakshy, E.
   *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.*
   NeurIPS 2020.
   [Paper](https://arxiv.org/abs/1910.06403)
   [Code](https://github.com/meta-pytorch/botorch)

7. <span id="ref-7"></span> Chew, R. W. T., Chen, Z., Hemachandra, A., Low, B. K. H.
   *BoLT: A Benchmark to Democratize Black-box Optimization Research for
   Expensive LLM Tasks.* arXiv:2605.17000, 2026.
   [Paper](https://arxiv.org/abs/2605.17000)
   [Code](https://github.com/chewwt/bolt)
