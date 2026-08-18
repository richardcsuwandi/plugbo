# TuRBO plugin (region slot)

Switch with `lenz set-region --policy turbo`. Subsequent `suggest` (without `--bounds` / `--around`) optimizes acquisition inside a hyperrectangle trust region. The region is centered on the incumbent. Length doubles after a streak of improvements and halves after a streak of failures. Below a minimum length the region restarts.

You own *mode* (enable, disable, override). TuRBO owns the counters.

```bash
lenz turbo init --state ./state.json
lenz turbo status --state ./state.json
lenz turbo override --state ./state.json --length 0.4 --center '{"x": 0.3}'
lenz set-region --state ./state.json --policy box
```

Prefer this over improvised `set-bounds` when you want principled local search. Still use `suggest --around` for a one-shot local probe that does not persist.

Cite: Eriksson et al., NeurIPS 2019.
