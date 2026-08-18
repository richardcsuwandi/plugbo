#!/usr/bin/env bash
# Remaining work that is NOT already finished, in-flight, or queued by a
# live parent script (run_complement_backends.sh / run_bolt_lora_compare.sh).
#
# Those two parents already own the leftover cake / sara-only / bolt agent
# legs. This script will not launch a path they still intend to write.
# After they exit, rerun here to retry anything that is still missing or
# failed.
#
# Never-started RQ5 disclosure groups (branin / ackley / constrained /
# gp_sample extra backends) do not share log dirs with the live parents, so
# they are the only work this script will launch while those parents are up.
# They are off by default; pass --rq5 to include them.
#
# Usage:
#   ./scripts/run_unclaimed.sh           # list (default)
#   ./scripts/run_unclaimed.sh list
#   ./scripts/run_unclaimed.sh run
#   ./scripts/run_unclaimed.sh run --rq5
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_compare_env.sh

MODE="list"
RQ5=0
for arg in "$@"; do
  case "$arg" in
    list|run) MODE="$arg" ;;
    --rq5) RQ5=1 ;;
    -h|--help)
      echo "usage: $0 [list|run] [--rq5]" >&2
      exit 0
      ;;
    *)
      echo "usage: $0 [list|run] [--rq5]" >&2
      exit 1
      ;;
  esac
done

BUDGET="${BUDGET:-100}"
SEED="${SEED:-42}"
ACQF="${ACQF:-noisy_logei}"

plan_tsv="$(python3 - "$BUDGET" "$SEED" "$RQ5" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BUDGET, SEED, RQ5 = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
ROOT = Path("results/logs")


def ps_commands() -> list[str]:
    out = subprocess.check_output(["ps", "-ax", "-o", "command="], text=True)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


cmds = ps_commands()
complement_alive = any("run_complement_backends.sh run" in c for c in cmds)
bolt_alive = any(
    "run_bolt_lora_compare.sh" in c and "--list" not in c
    for c in cmds
)

# Paths those parents will still touch if they are alive.
claimed: dict[str, str] = {}
if complement_alive:
    for p in (
        "ackley10-compare/cake",
        "ackley10-compare/sara-only",
        "ackley20-compare/cake",
        "ackley20-compare/sara-only",
        "hartmann6-noblind-compare-3config-shifted/cake",
        "hartmann6-noblind-compare-3config-shifted/sara-only",
        "bolt_lora-compare/cake",
        "bolt_lora-compare/sara-only",
    ):
        claimed[p] = "complement"
if bolt_alive:
    for p in (
        "bolt_lora-compare/vanilla",
        "bolt_lora-compare/cake",
        "bolt_lora-compare/sara-lenz",
        "bolt_lora-compare/sara-lenz-cake",
        "bolt_lora-compare/sara-only",
    ):
        claimed[p] = "bolt" if p not in claimed else "complement+bolt"


def meta_status(cond: Path) -> tuple[str, int | None]:
    metas = list(cond.glob("sandbox_*/run_meta.json"))
    if not metas:
        return "missing", None
    latest = max(metas, key=lambda p: p.stat().st_mtime)
    status = json.loads(latest.read_text()).get("status") or "unknown"
    n_obs = None
    state = latest.parent / "state.json"
    if state.is_file():
        try:
            st = json.loads(state.read_text())
            n_obs = sum(1 for t in st.get("trials", []) if t.get("status") == "observed")
        except Exception:
            pass
    return status, n_obs


def emit(kind: str, path: str, note: str, cmd: str = "") -> None:
    print(f"{kind}\t{path}\t{note}\t{cmd}")


# Backend holes we still care about filling (same as complement + bolt remainder).
backend_rows = [
    ("ackley10-compare/cake", "blind cake-only"),
    ("ackley10-compare/sara-only", "blind sara-only"),
    ("ackley20-compare/cake", "blind cake-only"),
    ("ackley20-compare/sara-only", "blind sara-only"),
    ("hartmann6-noblind-compare-3config-shifted/cake", "revealed+shifted cake-only"),
    ("hartmann6-noblind-compare-3config-shifted/sara-only", "revealed+shifted sara-only"),
    ("bolt_lora-compare/sara-lenz", "revealed sara+lenz"),
    ("bolt_lora-compare/sara-lenz-cake", "revealed sara+lenz+cake"),
    ("bolt_lora-compare/sara-only", "revealed sara-only"),
]

# Finished siblings, for the inventory (never launched by this script).
finished_context = [
    "hartmann6-compare/vanilla",
    "hartmann6-compare/cake",
    "hartmann6-compare/sara-lenz",
    "hartmann6-compare/sara-lenz-cake",
    "hartmann6-compare/sara-only",
    "ackley10-compare/vanilla",
    "ackley10-compare/sara-lenz",
    "ackley10-compare/sara-lenz-cake",
    "ackley20-compare/vanilla",
    "ackley20-compare/sara-lenz",
    "ackley20-compare/sara-lenz-cake",
    "bolt_lora-compare/vanilla",
    "bolt_lora-compare/cake",
    "hartmann6-noblind-compare-3config-shifted/vanilla",
    "hartmann6-noblind-compare-3config-shifted/sara-lenz",
    "hartmann6-noblind-compare-3config-shifted/sara-lenz-cake",
]

for rel in finished_context:
    st, n = meta_status(ROOT / rel)
    extra = f" n_obs={n}" if n is not None else ""
    emit("FINISHED", rel, f"{st}{extra}")

for rel, label in backend_rows:
    st, n = meta_status(ROOT / rel)
    extra = f" n_obs={n}" if n is not None else ""
    owner = claimed.get(rel)
    if st in ("completed",):
        emit("FINISHED", rel, f"{st}{extra} {label}")
    elif st == "running":
        who = f" parent={owner}" if owner else ""
        emit("ONGOING", rel, f"{st}{extra}{who} {label}")
    elif owner:
        emit("QUEUED", rel, f"{st}{extra} parent={owner} {label}")
    else:
        emit("UNCLAIMED", rel, f"{st}{extra} {label}")

rq5 = [
    (
        "branin-noblind-compare",
        f"./scripts/run_noblind_compare.sh branin {BUDGET} {SEED} --root ./results/logs/branin-noblind-compare",
    ),
    (
        "ackley10-noblind-compare",
        f"./scripts/run_noblind_compare.sh ackley10 {BUDGET} {SEED} --root ./results/logs/ackley10-noblind-compare",
    ),
    (
        "ackley20-noblind-compare",
        f"./scripts/run_noblind_compare.sh ackley20 {BUDGET} {SEED} --root ./results/logs/ackley20-noblind-compare",
    ),
    (
        "constrained_hartmann6-noblind-compare",
        f"./scripts/run_noblind_compare.sh constrained_hartmann6 {BUDGET} {SEED} --root ./results/logs/constrained_hartmann6-noblind-compare",
    ),
    (
        "gp_sample6-noblind-compare-vanilla",
        f"./scripts/run_noblind_compare.sh gp_sample6 {BUDGET} {SEED} --config vanilla --root ./results/logs/gp_sample6-noblind-compare-vanilla",
    ),
    (
        "gp_sample6-noblind-compare-cake",
        f"./scripts/run_noblind_compare.sh gp_sample6 {BUDGET} {SEED} --config sara-lenz-cake --root ./results/logs/gp_sample6-noblind-compare-cake",
    ),
]

for rel, cmd in rq5:
    d = ROOT / rel
    if not d.exists():
        emit("RQ5", rel, "never started", cmd)
        continue
    statuses = []
    for child in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
        st, n = meta_status(child)
        statuses.append(f"{child.name}:{st}")
    note = ", ".join(statuses) if statuses else "empty dir"
    if any(s.endswith(":running") for s in statuses):
        emit("ONGOING", rel, note)
    elif statuses and all(":completed" in s for s in statuses):
        emit("FINISHED", rel, note)
    else:
        emit("RQ5", rel, note, cmd)

print(f"# complement_alive={int(complement_alive)} bolt_alive={int(bolt_alive)}", file=sys.stderr)
PY
)"

echo "mode=$MODE rq5=$RQ5 budget=$BUDGET seed=$SEED"
echo "This script does not touch paths owned by live complement / bolt parents."
echo

print_section() {
  local title="$1" kind="$2"
  local rows
  rows=$(printf '%s\n' "$plan_tsv" | awk -F '\t' -v k="$kind" '$1==k {print}')
  echo "=== $title ==="
  if [ -z "$rows" ]; then
    echo "  (none)"
    echo
    return 0
  fi
  while IFS=$'\t' read -r _ path note cmd; do
    printf '  %-56s %s\n' "$path" "$note"
  done <<< "$rows"
  echo
}

print_section "FINISHED (leave them)" FINISHED
print_section "ONGOING (leave them; live python job)" ONGOING
print_section "QUEUED (owned by a live parent; do not launch)" QUEUED
print_section "UNCLAIMED backend leftovers (this script will run these)" UNCLAIMED
print_section "RQ5 never-started disclosure groups (only with --rq5)" RQ5

run_unclaimed() {
  local rows
  rows=$(printf '%s\n' "$plan_tsv" | awk -F '\t' '$1=="UNCLAIMED" {print}')
  if [ -z "$rows" ]; then
    echo "No unclaimed backend leftovers. Complement / bolt already cover them."
    return 0
  fi
  while IFS=$'\t' read -r _ path note _; do
    echo "RUN backend leftover: $path  ($note)"
    case "$path" in
      ackley10-compare/cake)
        python3 -m benchmarks.run_blind_baseline --benchmark ackley10 --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/ackley10-compare/cake --policy cake \
          --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
          --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
          --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/ackley10-compare --title "ackley10 blind comparison"
        ;;
      ackley10-compare/sara-only)
        python3 -m benchmarks.run_blind_test --benchmark ackley10 --provider "$PROVIDER" --model "$MODEL" \
          --base-url "$BASE_URL" --api-key "$API_KEY" --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/ackley10-compare/sara-only --no-lenz --extra-body "$EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/ackley10-compare --title "ackley10 blind comparison"
        ;;
      ackley20-compare/cake)
        python3 -m benchmarks.run_blind_baseline --benchmark ackley20 --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/ackley20-compare/cake --policy cake \
          --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
          --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
          --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/ackley20-compare --title "ackley20 blind comparison"
        ;;
      ackley20-compare/sara-only)
        python3 -m benchmarks.run_blind_test --benchmark ackley20 --provider "$PROVIDER" --model "$MODEL" \
          --base-url "$BASE_URL" --api-key "$API_KEY" --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/ackley20-compare/sara-only --no-lenz --extra-body "$EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/ackley20-compare --title "ackley20 blind comparison"
        ;;
      hartmann6-noblind-compare-3config-shifted/cake)
        python3 -m benchmarks.run_blind_baseline --benchmark hartmann6 --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/hartmann6-noblind-compare-3config-shifted/cake --policy cake --reveal --shift \
          --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
          --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
          --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/hartmann6-noblind-compare-3config-shifted \
          --title "hartmann6 no-blind comparison (shifted)"
        ;;
      hartmann6-noblind-compare-3config-shifted/sara-only)
        python3 -m benchmarks.run_noblind_test --benchmark hartmann6 --provider "$PROVIDER" --model "$MODEL" \
          --base-url "$BASE_URL" --api-key "$API_KEY" --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/hartmann6-noblind-compare-3config-shifted/sara-only \
          --no-lenz --shift --extra-body "$EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/hartmann6-noblind-compare-3config-shifted \
          --title "hartmann6 no-blind comparison (shifted)"
        ;;
      bolt_lora-compare/sara-lenz)
        python3 -m benchmarks.run_noblind_test --benchmark bolt_lora --provider "$PROVIDER" --model "$MODEL" \
          --base-url "$BASE_URL" --api-key "$API_KEY" --budget "$BUDGET" --seed "$SEED" --warmup 8 \
          --root ./results/logs/bolt_lora-compare/sara-lenz --surrogate fixed --acqf "$ACQF" \
          --extra-body "$EXTRA_BODY"
        ;;
      bolt_lora-compare/sara-lenz-cake)
        python3 -m benchmarks.run_noblind_test --benchmark bolt_lora --provider "$PROVIDER" --model "$MODEL" \
          --base-url "$BASE_URL" --api-key "$API_KEY" --budget "$BUDGET" --seed "$SEED" --warmup 8 \
          --root ./results/logs/bolt_lora-compare/sara-lenz-cake --surrogate cake --acqf "$ACQF" \
          --extra-body "$EXTRA_BODY" \
          --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
          --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
          --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
        ;;
      bolt_lora-compare/sara-only)
        python3 -m benchmarks.run_noblind_test --benchmark bolt_lora --provider "$PROVIDER" --model "$MODEL" \
          --base-url "$BASE_URL" --api-key "$API_KEY" --budget "$BUDGET" --seed "$SEED" \
          --root ./results/logs/bolt_lora-compare/sara-only --no-lenz --extra-body "$EXTRA_BODY"
        python3 -m benchmarks.plot_compare --root ./results/logs/bolt_lora-compare \
          --title "BOLT LoRA HPO (revealed names, matched warmup)"
        ;;
      *)
        echo "error: no command mapped for $path" >&2
        exit 1
        ;;
    esac
  done <<< "$rows"
}

run_rq5() {
  local rows
  rows=$(printf '%s\n' "$plan_tsv" | awk -F '\t' '$1=="RQ5" {print}')
  if [ -z "$rows" ]; then
    echo "No never-started RQ5 groups."
    return 0
  fi
  while IFS=$'\t' read -r _ path note cmd; do
    echo "RUN RQ5: $path  ($note)"
    eval "$cmd"
  done <<< "$rows"
}

if [ "$MODE" = "list" ]; then
  echo "Nothing launched."
  echo "  ./scripts/run_unclaimed.sh run        # only UNCLAIMED backend leftovers"
  echo "  ./scripts/run_unclaimed.sh run --rq5  # leftovers + never-started disclosure groups"
  exit 0
fi

run_unclaimed
if [ "$RQ5" = "1" ]; then
  echo
  run_rq5
fi

echo
echo "Refresh charts with: python3 -m benchmarks.plot_all"
