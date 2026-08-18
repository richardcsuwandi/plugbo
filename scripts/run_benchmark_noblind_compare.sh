#!/usr/bin/env bash
# The no-blind twin of run_benchmark_compare.sh: same three configs (vanilla
# BO / lenz only, sara + lenz, sara + lenz + cake), but every condition runs
# against a sandbox that reveals the benchmark's real identity, real
# parameter names, and real bounds (benchmarks.sandbox.build_sandbox(reveal=True)
# via run_blind_baseline.py --reveal / run_noblind_test.py) instead of hiding
# them. Useful for asking "given the identity, which backend configuration
# actually benefits from it" -- as opposed to run_noblind_compare.sh, which
# holds the backend fixed and instead varies how much is disclosed.
#
# By default the optimum is left at its exact textbook location (pure
# one-shot-recall test). Pass --shift-only to instead relocate it (same
# transform as the blind harness) for all three conditions -- identity is
# still revealed, but recall alone no longer solves it.
#
# Usage:
#   ./scripts/run_benchmark_noblind_compare.sh <benchmark> [budget] [seed] [--shift-only]
#
# Examples:
#   ./scripts/run_benchmark_noblind_compare.sh hartmann6
#   ./scripts/run_benchmark_noblind_compare.sh ackley10 100 42 --shift-only
#
# Same provider/model/creds override pattern as run_benchmark_compare.sh
# (defaults to ModelScope/Qwen via .env).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SHIFT_ONLY=0
ROOT_OVERRIDE=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --shift-only)
      SHIFT_ONLY=1
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

if [ -z "${1:-}" ]; then
  echo "usage: $0 <benchmark> [budget] [seed] [--shift-only] [--root PATH]" >&2
  echo "  <benchmark>: one of $(python3 -c 'from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))'), or gp_sample<dim> e.g. gp_sample6" >&2
  exit 1
fi
BENCHMARK="$1"
BUDGET="${2:-100}"
SEED="${3:-42}"
# Default root is keyed only by benchmark name, not --shift-only -- running
# both a shifted and an unshifted sweep for the same benchmark with the
# default root would land in the same vanilla/sara-lenz/sara-lenz-cake
# folders and comingle the two. Pass --root explicitly for the second one.
ROOT="${ROOT_OVERRIDE:-./results/logs/${BENCHMARK}-noblind-compare-3config}"
ACQF="${ACQF:-noisy_logei}"

source scripts/_compare_env.sh

SHIFT_FLAG=()
SHIFT_LABEL="no shift -- pure recall"
if [ "$SHIFT_ONLY" = "1" ]; then
  SHIFT_FLAG=(--shift)
  SHIFT_LABEL="shifted"
fi

echo "benchmark=$BENCHMARK budget=$BUDGET seed=$SEED acqf=$ACQF provider=$PROVIDER model=$MODEL base_url=${BASE_URL:-<default>} shift=$SHIFT_LABEL"
echo

# Never rm -rf here: $ROOT is keyed only by benchmark name, so two
# invocations (different seeds, or a concurrent rerun) share it. Each
# condition writes into its own freshly-tokened sandbox_<token>/ dir, so
# accumulating old runs alongside new ones is safe -- plot_compare.py already
# picks the latest sandbox per condition by mtime. Wiping the whole tree here
# previously deleted an unrelated run's in-flight state.json out from under it.
mkdir -p "$ROOT"

echo "=== [1/3] vanilla BO (no sara, no LLM), identity revealed ==="
python3 -m benchmarks.run_blind_baseline \
  --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$SEED" \
  --root "$ROOT/vanilla" --policy vanilla \
  --reveal "${SHIFT_FLAG[@]+"${SHIFT_FLAG[@]}"}"

echo
echo "=== [2/3] sara + lenz (cake off), identity revealed ==="
python3 -m benchmarks.run_noblind_test \
  --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-lenz" \
  --surrogate fixed --acqf "$ACQF" \
  --extra-body "$EXTRA_BODY" "${SHIFT_FLAG[@]+"${SHIFT_FLAG[@]}"}"

echo
echo "=== [3/3] sara + lenz + cake, identity revealed ==="
python3 -m benchmarks.run_noblind_test \
  --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-lenz-cake" \
  --surrogate cake --acqf "$ACQF" \
  --extra-body "$EXTRA_BODY" "${SHIFT_FLAG[@]+"${SHIFT_FLAG[@]}"}" \
  --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
  --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
  --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"

echo
echo "=== plotting comparison ==="
python3 -m benchmarks.plot_compare --root "$ROOT" --title "$BENCHMARK no-blind comparison ($SHIFT_LABEL)"

echo
echo "Open $ROOT/compare.html for the overlaid regret chart."
echo "Or browse each run's own trace/state: sara-viz --root $ROOT"
