# Contributing to PlugBO

Bug reports, plugins, and new benchmarks are welcome. By participating, you
agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

Open a GitHub issue first if the change adds a slot, a new experiment backend,
or a large dependency. Use the bug template for failures and the proposal
template for new methods or functions. Do not paste API keys or `.env`.

This repository is an independent re-implementation of Sara and lenz. The
plugin protocol is the main extension. Keep Sara's tools as `bash` and `read`.
New capability should appear as a `lenz` verb, not a second agent.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # only if you will call an LLM
```

Python 3.10+. Do not commit `.env`, `state.json`, or anything under `results/`.

---

## Tests

Run the default, no-network checks for the code you touched:

```bash
pytest -m "not slow"
./scripts/smoke_plugins.sh
```

`pytest` covers unit tests. The smoke script drives a short `lenz` CLI loop
so a plugin that unit-tests still has to work as a verb.

Extra smoke flags (`--live`, `--baseline`, and any later ones) are optional.
Use them only when your change hits that layer (LLM-backed plugins, the
scripted BO loop). The script header documents what each flag currently
does. `--live` needs API keys.

Add or extend tests next to the existing ones for that layer (`tests/`). A
plugin PR should prove occupancy and the hook. A benchmark PR should prove
the function values and that identity does not leak into agent-visible
files.

---

## Adding a BO plugin

A method occupies one slot of the live policy and stores its own blob under
`frame.plugins[<name>]`. Core never imports plugin internals. It calls hooks,
and the plugin registers CLI verbs.

| Slot | Default (no plugin) | Hook | Occupancy |
|---|---|---|---|
| `surrogate` | `fixed` | `on_observe` (and any model the slot owns) | `lenz set-surrogate --surrogate <name>` |
| `region` | `box` | `active_bounds`, often `on_observe` | `lenz set-region --policy <name>` |
| `prior` | `none` | `wrap_acqf` | a plugin verb that sets `shelf.prior` |
| `sampler` | `botorch` | `propose` | `lenz set-sampler --sampler <name>` |

Do not add a fifth slot without an issue. `lenz/optimize.py` and the shelf
fields are built around these four. Existing occupants in `lenz/plugins/`
are the worked examples.

### Files

1. `lenz/plugins/<name>.py`: subclass `LenzPlugin` from
   [`lenz/plugins/base.py`](lenz/plugins/base.py). Set `name` and `slot`.
2. `lenz/plugins/<name>.md`: short note for Sara (when to enable, verbs,
   failure modes). `prompt_path()` loads
   `lenz/plugins/{plugin.name}.md` by default. Sara concatenates every
   installed plugin file onto `SYSTEM.md` + `LENZ_REF.md`. Keep
   method-specific text out of
   [`sara/prompts/LENZ_REF.md`](sara/prompts/LENZ_REF.md).
3. Register an instance in `all_plugins()` in
   [`lenz/plugins/registry.py`](lenz/plugins/registry.py). Discovery is not
   automatic.

### Implementation rules

- Put method state in `self.blob(frame)` (`frame.plugins[name]`). Do not add
  fields to the live shelf.
- Occupying a slot must not discard trials.
- Extra verbs go through `add_parser` + `commands()`. Prefer namespaced
  verbs (`lenz <name> status`) over new global verbs when the method is
  optional.
- `on_observe` runs after a successful observe or submit-with-metrics, and
  only if this plugin occupies its slot.
- Return `None` from `active_bounds` / `propose` to fall back to the core
  default.

Minimal shape (any slot; region shown):

```python
from .base import SLOT_REGION, LenzPlugin

class MyRegionPlugin(LenzPlugin):
    name = "myregion"
    slot = SLOT_REGION

    def default_state(self) -> dict:
        return {}

    def add_parser(self, sub, state_parent) -> None:
        p = sub.add_parser("myregion", parents=[state_parent])
        inner = p.add_subparsers(dest="myregion_cmd", required=True)
        inner.add_parser("status")

    def commands(self):
        return {"myregion": self._dispatch}

    def active_bounds(self, frame, encoder):
        if frame.shelf.region != self.name:
            return None
        # (2, d) tensor in GP space, or None to use shelf.bounds
        ...
```

Then add the class to `all_plugins()`. After install:

```bash
lenz plugins --state ./state.json
lenz set-region --state ./state.json --policy myregion
```

Use the occupancy verb for the slot you filled (`set-surrogate`,
`set-region`, `set-sampler`, or the prior plugin's own verb).

### Tests

- Assert the new name appears in `lenz plugins` and occupies the expected
  slot.
- Unit-test the hook (bounds, reweighted acquisition, packed proposals, or
  whatever the slot does).
- If the plugin can run without an LLM, add a short CLI loop to the default
  smoke path.

### Optional: experiment backends

A plugin is usable from the CLI without this step. Wire it into comparison
scripts only if you want `./scripts/run_synthetic.sh --backend ...` (or the
mixed-type runner) to pin it.

- Scripted, no agent: a `Policy` in
  [`benchmarks/run_blind_baseline.py`](benchmarks/run_blind_baseline.py).
- Agent-pinned: argparse on
  [`benchmarks/run_blind_test.py`](benchmarks/run_blind_test.py) (and the
  revealed counterpart), then a `case` arm in
  [`scripts/_run_lib.sh`](scripts/_run_lib.sh).
- Viewer labels: [`viz/captions.py`](viz/captions.py) and a caption test.

Named beliefs, if the method needs them, belong with the other fixtures in
[`benchmarks/priors.py`](benchmarks/priors.py).

---

## Adding a benchmark

Functions live in [`benchmarks/functions.py`](benchmarks/functions.py). The
agent never sees that module. Blind runs go through
[`benchmarks/obfuscate.py`](benchmarks/obfuscate.py) (renamed parameters,
unit cube, shifted optimum) and
[`benchmarks/sandbox.py`](benchmarks/sandbox.py) (generic `context.md` plus a
hand-written oracle). Scoring uses a `secret.json` outside the sandbox.

### Closed-form function

1. Implement `fn(x: list[float]) -> float` and a `BenchmarkSpec` (`bounds`,
   `minimize`, known `f_opt`).
2. Register it in `REGISTRY`.
3. Add a generic oracle formula in `sandbox.py` (exact name, or a prefix for
   a family). That template is what the agent's `bash` tool can read: do not
   put the benchmark name or other identifying strings in it.
4. If there is a constraint, set `constraint_fn` / `constraint_upper` and
   add a matching constraint template.
5. Tests: known optimum matches `f_opt`, the name is registered, and
   identity does not leak into agent-visible sandbox files.

Then:

```bash
python3 -c "from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))"
./scripts/run_synthetic.sh <name> --backend vanilla --budget 20 --seed 42
```

Random GP sample paths (`gp_sample<dim>`) are not in `REGISTRY`. Copy that
path only for another no-memorization control.

### Mixed-type or learned oracle

Set `spec.space` to a lenz JSON search space and `allow_shift=False`. Follow
the surrogate-oracle branch in `sandbox.py`. Optional extra dependencies go
in `pyproject.toml`. Context variants, if you need them, follow the existing
mixed-type runner rather than the synthetic script.

A standalone demo that is not part of the harness can live under
`examples/`. See [`examples/branin/`](examples/branin/).

---

## Adding an example

`examples/<name>/` should contain:

- `context.md`: what Sara reads
- `eval.py`: `python3 eval.py '<config-json>'` prints a metrics JSON object
- optional `run_manual.py` for a no-agent lenz loop

Keep the eval script a black box. Do not ask the agent to open it.

---

## Pull requests

Fork the repo, branch from `main`, and open a pull request against `main`.
Fill in the PR template.

- One concern per PR (a plugin, a benchmark, or a harness fix).
- Match the surrounding style (type hints, no unused imports, wrap near 80
  columns).
- Update the plugin table in [`README.md`](README.md) when you add a slot
  occupant that you expect others to use.
- Do not check in campaign logs, API keys, or generated sandboxes.

Contributions are under the MIT license. See [LICENSE](LICENSE).
