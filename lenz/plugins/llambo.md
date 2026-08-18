# LLAMBO plugin (sampler slot)

Does not replace the GP. Use it to propose candidates from natural-language context and recent trials, then `score` them against lenz.

Configure a *separate* sampler LLM (same idea as CAKE's kernel LLM):

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
