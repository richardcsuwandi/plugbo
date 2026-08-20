#!/usr/bin/env bash
# Synthetic-function experiments (branin, hartmann6, ackley, gp_sampleN, …).
#
#   ./scripts/run_synthetic.sh hartmann6
#   ./scripts/run_synthetic.sh hartmann6 --disclosure revealed
#   ./scripts/run_synthetic.sh hartmann6 --disclosure revealed-shift
#   ./scripts/run_synthetic.sh hartmann6 --backend sara-lenz --disclosure all
#   ./scripts/run_synthetic.sh hartmann6 --backend turbo,vanilla --budget 30 --seed 42
#
# --disclosure  blind | revealed | revealed-shift | all
# --backend     comma list, or all (vanilla,sara-lenz,sara-lenz-cake)
# --warmup     shared Sobol evaluations (default: d+1 with a seed)
# Completed and in-flight legs are skipped; pass --force to rerun.
set -euo pipefail
# Interactive zsh stops a background job that writes to the terminal
# ("suspended (tty output)"). Ignore SIGTTOU/SIGTTIN so `cmd > log 2>&1 &`
# keeps running; children inherit this. Redirects still capture the logs.
trap '' TTOU TTIN
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DISCLOSURE="blind"
BACKEND="vanilla,sara-lenz,sara-lenz-cake"
ROOT_OVERRIDE=""
FORCE=0
LIST=0
BUDGET=100
SEED=42
WARMUP=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --disclosure)
      DISCLOSURE="${2:?}"
      shift 2
      ;;
    --backend)
      BACKEND="${2:?}"
      shift 2
      ;;
    --budget)
      BUDGET="${2:?}"
      shift 2
      ;;
    --seed)
      SEED="${2:?}"
      shift 2
      ;;
    --warmup)
      WARMUP="${2:?}"
      shift 2
      ;;
    --root)
      ROOT_OVERRIDE="${2:?}"
      shift 2
      ;;
    --force) FORCE=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

if [ -z "${1:-}" ]; then
  echo "usage: $0 <benchmark> [--disclosure blind|revealed|revealed-shift|all] [--backend LIST] [--budget N] [--seed N] [--warmup N]" >&2
  echo "  <benchmark>: one of $(python3 -c 'from benchmarks.functions import REGISTRY; print(sorted(REGISTRY))'), or gp_sample<dim>" >&2
  exit 1
fi
BENCHMARK="$1"

case "$DISCLOSURE" in
  blind|revealed|revealed-shift|all) ;;
  *)
    echo "error: --disclosure must be blind, revealed, revealed-shift, or all" >&2
    exit 1
    ;;
esac

source scripts/_compare_env.sh
export FORCE LIST BUDGET SEED WARMUP
source scripts/_run_lib.sh

BACKENDS="$(expand_backends "$BACKEND" "vanilla,sara-lenz,sara-lenz-cake")"
# shellcheck disable=SC2206
BACKEND_ARR=($BACKENDS)
N_BACKENDS=${#BACKEND_ARR[@]}

if [ "$DISCLOSURE" = "all" ] && [ "$N_BACKENDS" -ne 1 ]; then
  echo "error: --disclosure all needs exactly one --backend (got: $BACKENDS)" >&2
  exit 1
fi

echo "benchmark=$BENCHMARK budget=$BUDGET seed=$SEED warmup=${WARMUP:-auto} disclosure=$DISCLOSURE backend=$BACKENDS provider=$PROVIDER model=$MODEL"
echo

if [ "$DISCLOSURE" = "all" ]; then
  backend="${BACKEND_ARR[0]}"
  ROOT="${ROOT_OVERRIDE:-./results/logs/${BENCHMARK}-disclosure-${backend}}"
  mkdir -p "$ROOT"
  run_leg "$ROOT/blind" "$BENCHMARK" "$backend" blind
  echo
  run_leg "$ROOT/revealed-shift" "$BENCHMARK" "$backend" revealed-shift
  echo
  run_leg "$ROOT/revealed" "$BENCHMARK" "$backend" revealed
  echo
  plot_root "$ROOT" "$BENCHMARK disclosure sweep ($backend)"
else
  ROOT="${ROOT_OVERRIDE:-./results/logs/${BENCHMARK}-${DISCLOSURE}}"
  mkdir -p "$ROOT"
  for backend in "${BACKEND_ARR[@]}"; do
    run_leg "$ROOT/$backend" "$BENCHMARK" "$backend" "$DISCLOSURE"
    echo
  done
  plot_root "$ROOT" "$BENCHMARK $DISCLOSURE"
fi

echo
echo "Open $ROOT/compare.html  (or plugbo-viz --root $ROOT)"
