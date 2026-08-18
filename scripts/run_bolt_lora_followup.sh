#!/usr/bin/env bash
# Follow-up on bolt_lora-compare (seed 42, domain context, already finished):
#
#   Wave 1  context ablation at seed 42: generic + misleading, Sara+lenz and
#           Sara-only only. Vanilla from bolt_lora-compare is reused (it never
#           reads context.md). Does not touch the finished domain group.
#   Wave 2  extra seeds 7 and 13 on the domain prompt: vanilla, Sara+lenz,
#           Sara-only. Together with seed 42 that is three seeds.
#   Wave 3  only if wave 2 is still ambiguous: seeds 5 and 11, same three
#           methods. No CAKE (add later by hand if you still need it).
#
# One script, three waves. Default is list. New --root dirs, so this does not
# collide with bolt_lora-compare or with complement's remaining Ackley/Hartmann
# legs. It does share the ModelScope endpoint: refuse if complement is still
# live unless you pass --ignore-live.
#
# Usage:
#   ./scripts/run_bolt_lora_followup.sh           # list (default)
#   ./scripts/run_bolt_lora_followup.sh list
#   ./scripts/run_bolt_lora_followup.sh 1
#   ./scripts/run_bolt_lora_followup.sh 2
#   ./scripts/run_bolt_lora_followup.sh 3
#   ./scripts/run_bolt_lora_followup.sh all        # wave 1 then 2 (not 3)
#   ./scripts/run_bolt_lora_followup.sh 1 --ignore-live
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_compare_env.sh

MODE="list"
IGNORE_LIVE=0
for arg in "$@"; do
  case "$arg" in
    list|1|2|3|all) MODE="$arg" ;;
    --ignore-live) IGNORE_LIVE=1 ;;
    -h|--help)
      echo "usage: $0 [list|1|2|3|all] [--ignore-live]" >&2
      exit 0
      ;;
    *)
      echo "usage: $0 [list|1|2|3|all] [--ignore-live]" >&2
      exit 1
      ;;
  esac
done

BUDGET="${BUDGET:-100}"
WARMUP="${WARMUP:-8}"
ACQF="${ACQF:-noisy_logei}"
BENCHMARK="bolt_lora"

complement_alive() {
  ps -ax -o command= | grep -F "run_complement_backends.sh run" | grep -v grep >/dev/null
}

if [ "$MODE" != "list" ] && complement_alive && [ "$IGNORE_LIVE" != "1" ]; then
  echo "error: run_complement_backends.sh run is still live (ModelScope load)." >&2
  echo "Wait for it, or rerun with --ignore-live if you accept the extra traffic." >&2
  echo "Preview only: $0 list" >&2
  exit 1
fi

echo "mode=$MODE budget=$BUDGET warmup=$WARMUP acqf=$ACQF provider=$PROVIDER model=$MODEL"
echo "wave 1: generic + misleading at seed 42 (sara-lenz, sara-only)"
echo "wave 2: domain prompt, seeds 7 and 13 (vanilla, sara-lenz, sara-only)"
echo "wave 3: domain prompt, seeds 5 and 11 (same three methods; optional)"
echo "finished domain seed-42 group is left alone: results/logs/bolt_lora-compare"
echo

run_or_skip() {
  local cond_dir="$1"
  local label="$2"
  shift 2
  local st
  st=$(condition_status "$cond_dir")
  case "$st" in
    completed|running)
      echo "SKIP  $label  ($st)"
      return 0
      ;;
  esac
  echo "RUN   $label  (status=$st)"
  if [ "$MODE" = "list" ]; then
    return 0
  fi
  mkdir -p "$cond_dir"
  "$@"
}

run_vanilla() {
  local root="$1" seed="$2"
  run_or_skip "$root/vanilla" "$root/vanilla seed=$seed" \
    python3 -m benchmarks.run_blind_baseline \
      --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$seed" --warmup "$WARMUP" \
      --root "$root/vanilla" --policy vanilla --reveal
}

run_sara_lenz() {
  local root="$1" seed="$2" variant="${3:-domain}"
  run_or_skip "$root/sara-lenz" "$root/sara-lenz seed=$seed variant=$variant" \
    python3 -m benchmarks.run_noblind_test \
      --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$seed" --warmup "$WARMUP" --root "$root/sara-lenz" \
      --surrogate fixed --acqf "$ACQF" \
      --context-variant "$variant" \
      --extra-body "$EXTRA_BODY"
}

run_sara_only() {
  local root="$1" seed="$2" variant="${3:-domain}"
  run_or_skip "$root/sara-only" "$root/sara-only seed=$seed variant=$variant" \
    python3 -m benchmarks.run_noblind_test \
      --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$seed" --root "$root/sara-only" \
      --no-lenz --context-variant "$variant" --extra-body "$EXTRA_BODY"
}

plot_group() {
  local root="$1" title="$2"
  if [ "$MODE" = "list" ]; then
    echo "PLOT  python3 -m benchmarks.plot_compare --root $root"
    return 0
  fi
  python3 -m benchmarks.plot_compare --root "$root" --title "$title"
}

wave1() {
  echo "=== wave 1: context ablation, seed 42 (reuse vanilla from bolt_lora-compare) ==="
  local root v
  for v in generic misleading; do
    root="./results/logs/bolt_lora-${v}-compare"
    echo "--- variant=$v ---"
    run_sara_lenz "$root" 42 "$v"
    echo
    run_sara_only "$root" 42 "$v"
    echo
    plot_group "$root" "BOLT LoRA HPO ($v context, seed 42)"
    echo
  done
}

wave2() {
  echo "=== wave 2: domain context, seeds 7 and 13 ==="
  local seed root
  for seed in 7 13; do
    root="./results/logs/bolt_lora-seed${seed}-compare"
    echo "--- seed=$seed ---"
    run_vanilla "$root" "$seed"
    echo
    run_sara_lenz "$root" "$seed" domain
    echo
    run_sara_only "$root" "$seed" domain
    echo
    plot_group "$root" "BOLT LoRA HPO (domain context, seed $seed)"
    echo
  done
}

wave3() {
  echo "=== wave 3: domain context, seeds 5 and 11 (only if wave 2 is still ambiguous) ==="
  local seed root
  for seed in 5 11; do
    root="./results/logs/bolt_lora-seed${seed}-compare"
    echo "--- seed=$seed ---"
    run_vanilla "$root" "$seed"
    echo
    run_sara_lenz "$root" "$seed" domain
    echo
    run_sara_only "$root" "$seed" domain
    echo
    plot_group "$root" "BOLT LoRA HPO (domain context, seed $seed)"
    echo
  done
}

case "$MODE" in
  list)
    MODE=list
    wave1
    wave2
    echo "Wave 3 is optional. Inspect wave 2, then: $0 3"
    echo "Nothing launched."
    ;;
  1) MODE=1; wave1 ;;
  2) MODE=2; wave2 ;;
  3) MODE=3; wave3 ;;
  all)
    MODE=all
    wave1
    wave2
    echo "Wave 3 not launched. If seeds 7/13 still look like a coin flip: $0 3"
    ;;
esac

echo "Domain seed-42 reference (already finished): results/logs/bolt_lora-compare"
echo "Browse with: python3 -m viz.server"
