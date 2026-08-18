#!/usr/bin/env bash
# Complete sweep for bolt_lora (BOLT LoRA HPO emulator): a mixed-type
# regime-2 surface with domain language but no textbook optimum to recall.
# Shift is a no-op here, so the usual disclosure triangle
# (run_noblind_compare.sh) is the wrong experiment.
#
# Conditions, same seed / acqf / Sobol warm-start (d+1 = 8) except sara-nolen
# which has no lenz and therefore no warmup by default here (an explicit
# `--warmup N` on a `run_noblind_test.py --no-lenz` invocation now runs the
# same real Sobol warm-start its lenz-backed siblings get -- see
# benchmarks/run_noblind_test.py -- but this script leaves it off to match
# past runs; pass it yourself for a warmup-matched comparison):
#   1. vanilla BO          -- no Sara, fixed Matern
#   2. cake (no Sara)      -- scripted suggest/submit, CAKE kernel LLM only
#   3. sara + lenz         -- identity revealed, cake off
#   4. sara + lenz + cake  -- identity revealed, CAKE on
#   5. sara-only           -- LLM proposes every point; lenz is blocked
#
# (1) and (2) isolate covariance misspecification without an agent.
# (3) and (4) ask whether LoRA/Qwen domain language plus the agentic loop
# helps once the kernel question is already on the table. (5) asks whether
# the LLM can act as the optimizer with no BO backend.
# Vanilla/cake use --reveal so every condition sees the same native mixed
# space (real names, native int/choice types), not a renamed blind cube.
#
# Completed and in-flight legs are skipped, so a rerun after a crash is safe.
# Pass --force to ignore that and launch a fresh sandbox per selected leg.
#
# Usage:
#   ./scripts/run_bolt_lora_compare.sh [budget] [seed]
#   ./scripts/run_bolt_lora_compare.sh 100 42 --no-agent-only
#   ./scripts/run_bolt_lora_compare.sh 100 42 --agent-only
#   ./scripts/run_bolt_lora_compare.sh --list
#   ./scripts/run_bolt_lora_compare.sh --force
#
# --agent-only: Sara+lenz, Sara+lenz+cake, and Sara-only (skip vanilla/cake).
# --sara-only: deprecated alias for --agent-only (does NOT mean no-lenz).
#
# Defaults: budget 100, seed 42. Provider/model/creds from scripts/_compare_env.sh.
# Requires: pip install -e '.[bolt]' (Hugging Face download on first eval).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NO_AGENT_ONLY=0
AGENT_ONLY=0
FORCE=0
LIST=0
ROOT_OVERRIDE=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-agent-only)
      NO_AGENT_ONLY=1
      shift
      ;;
    --agent-only|--sara-only)
      AGENT_ONLY=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --list)
      LIST=1
      shift
      ;;
    --root)
      ROOT_OVERRIDE="${2:?--root requires a value}"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

if [ "$NO_AGENT_ONLY" = "1" ] && [ "$AGENT_ONLY" = "1" ]; then
  echo "error: pass only one of --no-agent-only / --agent-only" >&2
  exit 1
fi

BENCHMARK="bolt_lora"
BUDGET="${1:-100}"
SEED="${2:-42}"
ROOT="${ROOT_OVERRIDE:-./results/logs/${BENCHMARK}-compare}"
ACQF="${ACQF:-noisy_logei}"
# 7 mixed parameters -> lenz warmup threshold d+1. Shared across BO legs
# so post-warmup regret is comparable. Sara-only ignores this (no lenz).
WARMUP="${WARMUP:-8}"
PLOT_TITLE="BOLT LoRA HPO (revealed names, matched warmup)"

source scripts/_compare_env.sh

echo "benchmark=$BENCHMARK budget=$BUDGET seed=$SEED warmup=$WARMUP acqf=$ACQF provider=$PROVIDER model=$MODEL base_url=${BASE_URL:-<default>}"
echo "root=$ROOT force=$FORCE list=$LIST"
echo

mkdir -p "$ROOT"

run_or_skip() {
  local cond_dir="$1"
  local label="$2"
  shift 2
  local st
  st=$(condition_status "$cond_dir")
  if [ "$FORCE" != "1" ]; then
    case "$st" in
      completed|running)
        echo "SKIP  $label  ($st)"
        return 0
        ;;
    esac
  fi
  echo "RUN   $label  (status=$st)"
  if [ "$LIST" = "1" ]; then
    return 0
  fi
  mkdir -p "$cond_dir"
  "$@"
}

run_vanilla() {
  echo "=== vanilla BO, no sara, no LLM ==="
  run_or_skip "$ROOT/vanilla" vanilla \
    python3 -m benchmarks.run_blind_baseline \
      --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$SEED" --warmup "$WARMUP" \
      --root "$ROOT/vanilla" --policy vanilla --reveal
}

run_cake() {
  echo "=== cake, no sara, kernel LLM only ==="
  run_or_skip "$ROOT/cake" cake \
    python3 -m benchmarks.run_blind_baseline \
      --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$SEED" --warmup "$WARMUP" \
      --root "$ROOT/cake" --policy cake --reveal \
      --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
      --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
      --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
}

run_sara_lenz() {
  echo "=== sara + lenz, cake off, identity revealed ==="
  run_or_skip "$ROOT/sara-lenz" sara-lenz \
    python3 -m benchmarks.run_noblind_test \
      --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$SEED" --warmup "$WARMUP" --root "$ROOT/sara-lenz" \
      --surrogate fixed --acqf "$ACQF" \
      --extra-body "$EXTRA_BODY"
}

run_sara_lenz_cake() {
  echo "=== sara + lenz + cake, identity revealed ==="
  run_or_skip "$ROOT/sara-lenz-cake" sara-lenz-cake \
    python3 -m benchmarks.run_noblind_test \
      --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$SEED" --warmup "$WARMUP" --root "$ROOT/sara-lenz-cake" \
      --surrogate cake --acqf "$ACQF" \
      --extra-body "$EXTRA_BODY" \
      --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
      --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
      --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
}

run_sara_nolen() {
  echo "=== sara-only, no lenz, identity revealed ==="
  run_or_skip "$ROOT/sara-only" sara-only \
    python3 -m benchmarks.run_noblind_test \
      --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-only" \
      --no-lenz --extra-body "$EXTRA_BODY"
}

if [ "$AGENT_ONLY" != "1" ]; then
  run_vanilla
  echo
  run_cake
  echo
fi
if [ "$NO_AGENT_ONLY" != "1" ]; then
  run_sara_lenz
  echo
  run_sara_lenz_cake
  echo
  run_sara_nolen
  echo
fi

if [ "$LIST" = "1" ]; then
  echo "Nothing launched. Rerun without --list to execute the RUN lines."
  exit 0
fi

echo "=== plotting comparison ==="
python3 -m benchmarks.plot_compare --root "$ROOT" --title "$PLOT_TITLE"

echo
echo "Open $ROOT/compare.html for the overlaid regret chart."
echo "Or browse each run: sara-viz --root $ROOT"
