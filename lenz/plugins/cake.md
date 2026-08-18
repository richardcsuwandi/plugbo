# CAKE plugin (surrogate slot)

Switch with `lenz set-surrogate --surrogate cake`. Each objective and constraint metric gets its own kernel population. BAKER ranks weighted kernel combinations using the study's configured acquisition. If populations are not ready, lenz falls back to the best kernel per metric, or Sobol if none are fittable.

Kernel evolution uses Sara's LLM by default (`sara run --provider/--model`, or `lenz set-llm` / `lenz create --llm-*`). Pass `--kernel-llm-*` only to pick a different model:

```bash
lenz set-surrogate --state ./state.json --surrogate cake \
  --budget 30 \
  --kernel-llm-provider openai-compatible \
  --kernel-llm-model your-model-id \
  --kernel-llm-base-url https://api.example.com/v1 \
  --kernel-llm-api-key-env MY_KERNEL_LLM_API_KEY
```

Evolution runs on a schedule (after enough data, every few observations, frozen past a fraction of the budget). You do not need to trigger it. Force a round after a regime change:

```bash
lenz evolve-kernels --state ./state.json --force
lenz kernel-population --state ./state.json
```

Cite: Suwandi et al., NeurIPS 2025.
