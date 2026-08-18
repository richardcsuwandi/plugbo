#!/usr/bin/env bash
# Generalized version of run_hartmann6_compare.sh -- same blind
# anti-memorization three-way (sara+lenz, sara+lenz+cake, vanilla BO), same
# fixed acqf/seed/warm-start discipline, but for any benchmark in
# benchmarks.functions.REGISTRY (or a fresh gp_sample<dim> draw), so the
# paper's own suite (branin, hartmann6, ackley10, ackley20,
# constrained_hartmann6, gp_sample<dim>) and the extra textbook functions
# added alongside it (rosenbrock, rastrigin, levy, griewank, michalewicz,
# styblinski_tang, shekel, six_hump_camel) can all be run the same way
# without a hand-copied script per function.
#
# Companions: run_benchmark_noblind_compare.sh (same 3 configs, identity
# revealed) and run_noblind_compare.sh (one fixed config, three disclosure
# levels).
#
# Usage:
#   ./scripts/run_benchmark_compare.sh <benchmark> [budget] [seed]
#
# Examples:
#   ./scripts/run_benchmark_compare.sh rastrigin6
#   ./scripts/run_benchmark_compare.sh gp_sample6 60 7
#   ./scripts/run_benchmark_compare.sh constrained_hartmann6 100 42   # cake condition auto-skipped (unsupported for constrained studies)
#
# Defaults to ModelScope (openai-compatible endpoint) driving Qwen models --
# reads MODELSCOPE_API_KEY / MODELSCOPE_BASE_URL from .env at repo root.
# Override provider/model/creds via env vars, e.g.:
#   PROVIDER=anthropic MODEL=claude-opus-5 ./scripts/run_benchmark_compare.sh hartmann6
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${1:-}" ]; then
  echo "usage: $0 <benchmark> [budget] [seed]" >&2
  echo "  <benchmark>: one of $(python3 -c 'from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))'), or gp_sample<dim> e.g. gp_sample6" >&2
  exit 1
fi
BENCHMARK="$1"
BUDGET="${2:-100}"
SEED="${3:-42}"
ROOT="./results/logs/${BENCHMARK}-compare"
ACQF="${ACQF:-noisy_logei}"

source scripts/_compare_env.sh

HAS_CONSTRAINT="no"
is_constrained_benchmark "$BENCHMARK" && HAS_CONSTRAINT="yes"

echo "benchmark=$BENCHMARK budget=$BUDGET seed=$SEED acqf=$ACQF provider=$PROVIDER model=$MODEL base_url=${BASE_URL:-<default>} constrained=$HAS_CONSTRAINT"
echo

# Never rm -rf here: $ROOT is keyed only by benchmark name, so two
# invocations (different seeds, or a concurrent rerun) share it. Each
# condition writes into its own freshly-tokened sandbox_<token>/ dir, so
# accumulating old runs alongside new ones is safe -- plot_compare.py already
# picks the latest sandbox per condition by mtime. Wiping the whole tree here
# previously deleted an unrelated run's in-flight state.json out from under it.
mkdir -p "$ROOT"

echo "=== [1/3] sara + lenz (cake off) ==="
python3 -m benchmarks.run_blind_test \
  --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-lenz" \
  --surrogate fixed --acqf "$ACQF" \
  --extra-body "$EXTRA_BODY"

if [ "$HAS_CONSTRAINT" = "yes" ]; then
  echo
  echo "=== [2/3] sara + lenz + cake -- SKIPPED (cake doesn't support constrained studies) ==="
else
  echo
  echo "=== [2/3] sara + lenz + cake ==="
  python3 -m benchmarks.run_blind_test \
    --benchmark "$BENCHMARK" --provider "$PROVIDER" --model "$MODEL" \
    --base-url "$BASE_URL" --api-key "$API_KEY" \
    --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-lenz-cake" \
    --surrogate cake --acqf "$ACQF" \
    --extra-body "$EXTRA_BODY" \
    --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
    --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
    --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"
fi

echo
echo "=== [3/3] vanilla BO (no sara, no LLM) ==="
python3 -m benchmarks.run_blind_baseline \
  --benchmark "$BENCHMARK" --budget "$BUDGET" --seed "$SEED" \
  --root "$ROOT/vanilla" --policy vanilla

echo
echo "=== plotting comparison ==="
python3 -m benchmarks.plot_compare --root "$ROOT" --title "$BENCHMARK blind comparison"

echo
echo "Open $ROOT/compare.html for the overlaid regret chart."
echo "Or browse each run's own trace/state: sara-viz --root $ROOT"
