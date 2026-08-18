# LLAMBO plugin (sampler slot)

Does not replace the GP. Use it to propose candidates from natural-language context and recent trials, then `score` them against lenz.

Sampling uses Sara's LLM by default. Pass `--sampler-llm-*` or `llambo set-llm` only to pick a different model:

```bash
lenz llambo set-llm --state ./state.json --provider openai --model gpt-4.1
lenz llambo warmstart --state ./state.json --n 5 --context ./context.md
lenz llambo sample --state ./state.json --n 8
```

To make every `suggest` go through LLAMBO instead of BoTorch acquisition optimization:

```bash
lenz set-sampler --state ./state.json --sampler llambo
```

Strongest early in a run, when observations are scarce. After the GP is trustworthy, switch back to `botorch`.

Cite: Liu et al., ICLR 2024.
