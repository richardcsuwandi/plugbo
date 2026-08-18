#!/usr/bin/env bash
# Complement existing backend-compare groups with the two missing legs:
#
#   sara-only  -- LLM proposes every point; lenz is not created and cannot
#                 be imported or invoked (run_*_test --no-lenz)
#   cake       -- scripted BO, CAKE kernel LLM, no Sara
#                 (run_blind_baseline --policy cake)
#
# Only groups where those legs are a fair ablation (same budget/seed, same
# disclosure as siblings). Does NOT rerun completed or in-flight conditions.
# Failed conditions are retried (a new sandbox; plot_compare picks latest).
#
# Skip (adding a backend would confound the experiment):
#   hartmann6-noblind-compare, -vanilla, -cake
#   gp_sample6-noblind-compare
#     disclosure triangles: backend is the held-fixed factor
#   hartmann6-noblind-compare-3config
#     unshifted revealed Hartmann: the story is one-shot recall at eval 1.
#     cake-only would just be 100-eval BO; sara-lenz already one-shots.
#
# Default is list (print the plan and exact commands, run nothing).
#
# Usage:
#   ./scripts/run_complement_backends.sh              # list
#   ./scripts/run_complement_backends.sh list
#   ./scripts/run_complement_backends.sh run
#   ./scripts/run_complement_backends.sh run hartmann6-compare
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_compare_env.sh

BUDGET="${BUDGET:-100}"
SEED="${SEED:-42}"
ACQF="${ACQF:-noisy_logei}"
MODE="${1:-list}"
FILTER="${2:-}"

if [ "$MODE" != "list" ] && [ "$MODE" != "run" ]; then
  echo "usage: $0 [list|run] [group-name]" >&2
  exit 1
fi

condition_status() {
  python3 - "$1" <<'PY'
from pathlib import Path
import json, sys
d = Path(sys.argv[1])
if not d.is_dir():
    print("missing")
    raise SystemExit
metas = list(d.glob("sandbox_*/run_meta.json"))
if not metas:
    print("missing")
    raise SystemExit
latest = max(metas, key=lambda p: p.stat().st_mtime)
print(json.loads(latest.read_text()).get("status") or "unknown")
PY
}

print_cmd() {
  local out=() a redact=0
  for a in "$@"; do
    if [ "$redact" = 1 ]; then
      out+=('$API_KEY')
      redact=0
      continue
    fi
    [ "$a" = "--api-key" ] && redact=1
    out+=("$a")
  done
  printf '      '
  printf '%q ' "${out[@]}"
  echo
}

run_or_skip() {
  local cond_dir="$1"
  shift
  local st
  st=$(condition_status "$cond_dir")
  case "$st" in
    completed|running)
      echo "SKIP  $cond_dir  ($st)"
      return 0
      ;;
  esac
  echo "RUN   $cond_dir  (status=$st)"
  print_cmd "$@"
  if [ "$MODE" = "list" ]; then
    return 0
  fi
  mkdir -p "$cond_dir"
  "$@"
}

want_group() {
  local g="$1"
  [ -z "$FILTER" ] || [ "$FILTER" = "$g" ]
}

sara_only_blind() {
  local bench="$1" root="$2"
  run_or_skip "$root/sara-only" \
    python3 -m benchmarks.run_blind_test \
      --benchmark "$bench" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$SEED" --root "$root/sara-only" \
      --no-lenz --extra-body "$EXTRA_BODY"
}

cake_only_blind() {
  local bench="$1" root="$2"
  run_or_skip "$root/cake" \
    python3 -m benchmarks.run_blind_baseline \
      --benchmark "$bench" --budget "$BUDGET" --seed "$SEED" \
      --root "$root/cake" --policy cake \
      --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
      --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
      --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
}

sara_only_revealed() {
  local bench="$1" root="$2"
  shift 2
  run_or_skip "$root/sara-only" \
    python3 -m benchmarks.run_noblind_test \
      --benchmark "$bench" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$SEED" --root "$root/sara-only" \
      --no-lenz --extra-body "$EXTRA_BODY" "$@"
}

cake_only_revealed() {
  local bench="$1" root="$2"
  shift 2
  run_or_skip "$root/cake" \
    python3 -m benchmarks.run_blind_baseline \
      --benchmark "$bench" --budget "$BUDGET" --seed "$SEED" \
      --root "$root/cake" --policy cake --reveal \
      --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
      --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
      --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY" "$@"
}

plot_group() {
  local root="$1" title="$2"
  if [ "$MODE" = "list" ]; then
    echo "PLOT  python3 -m benchmarks.plot_compare --root $root --title $(printf '%q' "$title")"
    return 0
  fi
  python3 -m benchmarks.plot_compare --root "$root" --title "$title"
}

echo "mode=$MODE budget=$BUDGET seed=$SEED provider=$PROVIDER model=$MODEL"
echo "sara-only: LLM optimizer, no lenz. cake: kernel LLM, no Sara."
echo "skip completed/running. do not touch disclosure triangles or unshifted 3config."
echo

if want_group hartmann6-compare; then
  echo "=== hartmann6-compare (blind backend sweep) ==="
  cake_only_blind hartmann6 ./results/logs/hartmann6-compare
  sara_only_blind hartmann6 ./results/logs/hartmann6-compare
  plot_group ./results/logs/hartmann6-compare "hartmann6 blind comparison"
  echo
fi

if want_group ackley10-compare; then
  echo "=== ackley10-compare (blind backend sweep) ==="
  cake_only_blind ackley10 ./results/logs/ackley10-compare
  sara_only_blind ackley10 ./results/logs/ackley10-compare
  plot_group ./results/logs/ackley10-compare "ackley10 blind comparison"
  echo
fi

if want_group ackley20-compare; then
  echo "=== ackley20-compare (blind backend sweep) ==="
  cake_only_blind ackley20 ./results/logs/ackley20-compare
  sara_only_blind ackley20 ./results/logs/ackley20-compare
  plot_group ./results/logs/ackley20-compare "ackley20 blind comparison"
  echo
fi

if want_group hartmann6-noblind-compare-3config-shifted; then
  echo "=== hartmann6-noblind-compare-3config-shifted (revealed + shifted; search still required) ==="
  cake_only_revealed hartmann6 ./results/logs/hartmann6-noblind-compare-3config-shifted --shift
  sara_only_revealed hartmann6 ./results/logs/hartmann6-noblind-compare-3config-shifted --shift
  plot_group ./results/logs/hartmann6-noblind-compare-3config-shifted "hartmann6 no-blind comparison (shifted)"
  echo
fi

if want_group bolt_lora-compare; then
  echo "=== bolt_lora-compare (revealed mixed-type HPO; cake may already be in flight) ==="
  cake_only_revealed bolt_lora ./results/logs/bolt_lora-compare --warmup 8
  sara_only_revealed bolt_lora ./results/logs/bolt_lora-compare
  plot_group ./results/logs/bolt_lora-compare "BOLT LoRA HPO (revealed names, matched warmup)"
  echo
fi

echo "Skipped on purpose (not a backend ablation, or one-shot-recall confounding):"
echo "  hartmann6-noblind-compare, hartmann6-noblind-compare-vanilla, hartmann6-noblind-compare-cake"
echo "  gp_sample6-noblind-compare"
echo "  hartmann6-noblind-compare-3config  (unshifted; eval-1 recall, not a 100-eval backend sweep)"
echo
if [ "$MODE" = "list" ]; then
  echo "Nothing launched. To run the RUN lines above:"
  echo "  ./scripts/run_complement_backends.sh run"
  echo "  ./scripts/run_complement_backends.sh run hartmann6-compare"
else
  echo "Done. Refresh every compare.html with:"
  echo "  python3 -m benchmarks.plot_all"
fi
