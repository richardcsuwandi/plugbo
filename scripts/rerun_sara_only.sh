#!/usr/bin/env bash
# Rerun only the Sara-only leg with the shared seeded Sobol warm-start.
#
#   ./scripts/rerun_sara_only.sh hartmann6
#   ./scripts/rerun_sara_only.sh ackley10 --budget 100 --seed 42
#   ./scripts/rerun_sara_only.sh hartmann6 --warmup 7
#
# Extra arguments are forwarded to run_synthetic.sh. By default, the fresh
# run is written under results/logs/<benchmark>-compare.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -lt 1 ]; then
  echo "usage: $0 <benchmark> [run_synthetic.sh options]" >&2
  exit 2
fi

BENCHMARK="$1"
shift
ROOT="./results/logs/${BENCHMARK}-compare"

exec ./scripts/run_synthetic.sh "$BENCHMARK" \
  --backend sara-only \
  --disclosure blind \
  --root "$ROOT" \
  --force \
  "$@"
