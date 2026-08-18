#!/usr/bin/env bash
# NOT wired into any CI/make target and NOT meant to be run automatically --
# this spends real LLM budget probing memorization, run it deliberately.
#
# Runs one FIXED backend config through three identity-disclosure conditions
# (the opposite axis from run_benchmark_noblind_compare.sh, which holds
# disclosure fixed and varies the backend):
#
#   1. blind        -- benchmarks.run_blind_test (renamed params, unit cube,
#                       shifted optimum, identity hidden). The baseline.
#   2. noblind-shift -- identity revealed, real bounds/names, but the
#                       optimum IS relocated (same transform as blind). Can
#                       the model use recalled structure to accelerate a
#                       search it still has to do?
#   3. noblind       -- identity revealed, real bounds/names, NO shift. Pure
#                       one-shot-recall probe: can the model submit (close
#                       to) the textbook optimum as its very first move?
#
# A model that's purely pattern-matching benchmark names rather than
# optimizing should show: condition 3's evaluation-#1 regret near zero,
# condition 2 converging much faster than condition 1 despite an identical
# search problem, and near-zero across-the-board gaps disappearing entirely
# on a gp_sample<dim> benchmark (nothing to recall -- see
# docs/observations.md for why that control matters and what
# other benchmarks are worth adding this kind of probe to).
#
# --config selects the single backend used for all three conditions:
#   sara-lenz       (default) -- sara + lenz, cake off
#   sara-lenz-cake  -- sara + lenz + cake
#   vanilla         -- lenz only, no LLM. Revealing identity is a no-op for
#                      a non-agentic policy (it never reads context.md), so
#                      only the shift matters here -- this condition mainly
#                      exists as a memorization-free floor to compare the
#                      sara conditions against, not as a probe in its own
#                      right.
#
# Usage:
#   ./scripts/run_noblind_compare.sh <benchmark> [budget] [seed] [--config sara-lenz|sara-lenz-cake|vanilla]
#
# Examples:
#   ./scripts/run_noblind_compare.sh hartmann6
#   ./scripts/run_noblind_compare.sh ackley10 100 42 --config sara-lenz-cake
#   ./scripts/run_noblind_compare.sh hartmann6 100 42 --config vanilla
#
# Same provider/model/creds override pattern as run_benchmark_compare.sh
# (defaults to ModelScope/Qwen via .env).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="sara-lenz"
ROOT_OVERRIDE=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:?--config requires a value}"
      shift 2
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

case "$CONFIG" in
  sara-lenz|sara-lenz-cake|vanilla) ;;
  *)
    echo "error: --config must be one of sara-lenz, sara-lenz-cake, vanilla (got '$CONFIG')" >&2
    exit 1
    ;;
esac

if [ -z "${1:-}" ]; then
  echo "usage: $0 <benchmark> [budget] [seed] [--config sara-lenz|sara-lenz-cake|vanilla] [--root PATH]" >&2
  echo "  <benchmark>: one of $(python3 -c 'from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))') -- a gp_sample<dim> benchmark is a valid negative control (nothing to recall) but not a memorization probe" >&2
  exit 1
fi
BENCHMARK="$1"
BUDGET="${2:-100}"
SEED="${3:-42}"
# Default root is keyed only by benchmark name, not --config -- running a
# second --config against the same benchmark with the default root would
# write into the *same* blind/noblind-shift/noblind folders as an earlier
# config's run, comingling two different backends' sandboxes there. Pass
# --root explicitly (e.g. "${BENCHMARK}-noblind-compare-vanilla") whenever
# you run more than one --config for the same benchmark.
ROOT="${ROOT_OVERRIDE:-./results/logs/${BENCHMARK}-noblind-compare}"
ACQF="${ACQF:-noisy_logei}"
ONE_SHOT_TOL="${ONE_SHOT_TOL:-0.01}"

source scripts/_compare_env.sh

echo "benchmark=$BENCHMARK budget=$BUDGET seed=$SEED acqf=$ACQF config=$CONFIG provider=$PROVIDER model=$MODEL base_url=${BASE_URL:-<default>}"
echo

# Never rm -rf here: $ROOT is keyed only by benchmark name, so two
# invocations (different seeds, or a concurrent rerun) share it. Each
# condition writes into its own freshly-tokened sandbox_<token>/ dir, so
# accumulating old runs alongside new ones is safe -- plot_compare.py already
# picks the latest sandbox per condition by mtime. Wiping the whole tree here
# previously deleted an unrelated run's in-flight state.json out from under it.
mkdir -p "$ROOT"

run_blind() {
  local out="$1"
  case "$CONFIG" in
    vanilla)
      python3 -m benchmarks.run_blind_baseline \
        --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$SEED" \
        --root "$out" --policy vanilla
      ;;
    sara-lenz)
      python3 -m benchmarks.run_blind_test \
        --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
        --base-url "$BASE_URL" --api-key "$API_KEY" \
        --budget "$BUDGET" --seed "$SEED" --root "$out" \
        --surrogate fixed --acqf "$ACQF" --extra-body "$EXTRA_BODY"
      ;;
    sara-lenz-cake)
      python3 -m benchmarks.run_blind_test \
        --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
        --base-url "$BASE_URL" --api-key "$API_KEY" \
        --budget "$BUDGET" --seed "$SEED" --root "$out" \
        --surrogate cake --acqf "$ACQF" --extra-body "$EXTRA_BODY" \
        --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
        --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
        --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
      ;;
  esac
}

run_noblind() {
  local out="$1"
  local shift_flag=()
  [ "$2" = "shift" ] && shift_flag=(--shift)
  case "$CONFIG" in
    vanilla)
      python3 -m benchmarks.run_blind_baseline \
        --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$SEED" \
        --root "$out" --policy vanilla --reveal "${shift_flag[@]+"${shift_flag[@]}"}"
      ;;
    sara-lenz)
      python3 -m benchmarks.run_noblind_test \
        --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
        --base-url "$BASE_URL" --api-key "$API_KEY" \
        --budget "$BUDGET" --seed "$SEED" --root "$out" \
        --surrogate fixed --acqf "$ACQF" --extra-body "$EXTRA_BODY" \
        --one-shot-tol "$ONE_SHOT_TOL" "${shift_flag[@]+"${shift_flag[@]}"}"
      ;;
    sara-lenz-cake)
      python3 -m benchmarks.run_noblind_test \
        --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
        --base-url "$BASE_URL" --api-key "$API_KEY" \
        --budget "$BUDGET" --seed "$SEED" --root "$out" \
        --surrogate cake --acqf "$ACQF" --extra-body "$EXTRA_BODY" \
        --one-shot-tol "$ONE_SHOT_TOL" "${shift_flag[@]+"${shift_flag[@]}"}" \
        --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
        --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
        --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
      ;;
  esac
}

echo "=== [1/3] blind (anti-memorization baseline) -- config=$CONFIG ==="
run_blind "$ROOT/blind"

echo
echo "=== [2/3] no-blind, shifted (revealed identity, still has to search) -- config=$CONFIG ==="
run_noblind "$ROOT/noblind-shift" shift

echo
echo "=== [3/3] no-blind, no shift (pure one-shot-recall probe) -- config=$CONFIG ==="
run_noblind "$ROOT/noblind" noshift

echo
echo "=== plotting comparison ==="
python3 -m benchmarks.plot_compare --root "$ROOT" --title "$BENCHMARK blind vs. no-blind ($CONFIG)"

echo
echo "Open $ROOT/compare.html for the overlaid regret chart."
echo "Look at each condition's run_meta.json / stdout for the evaluation-#1 regret and one_shot_success flag (sara configs only)."
