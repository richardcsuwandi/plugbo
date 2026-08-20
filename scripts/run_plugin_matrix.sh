#!/usr/bin/env bash
# Expand a named local plugin experiment stage into paired benchmark runs.
# Local experiment driver: intentionally not part of the committed harness.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -lt 1 ]; then
  echo "usage: $0 <smoke|phase1|phase2|phase3|phase4> [--seeds CSV] [--conditions CSV] [--budget N] [--root DIR] [--list] [--force]" >&2
  exit 2
fi

STAGE="$1"
shift
SEEDS="42"
CONDITIONS=""
BUDGET=""
ROOT="./results/logs/plugin-matrix"
LIST=0
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds|--seed) SEEDS="${2:?}"; shift 2 ;;
    --conditions) CONDITIONS="${2:?}"; shift 2 ;;
    --budget) BUDGET="${2:?}"; shift 2 ;;
    --root) ROOT="${2:?}"; shift 2 ;;
    --list) LIST=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  smoke) DEFAULT_BUDGET=25 ;;
  phase1|phase2|phase3|phase4) DEFAULT_BUDGET=100 ;;
  *) echo "error: unknown stage '$STAGE'" >&2; exit 2 ;;
esac
BUDGET="${BUDGET:-$DEFAULT_BUDGET}"
if [ "$STAGE" = "smoke" ] && [ "$BUDGET" -lt 22 ]; then
  echo "error: smoke budget must be at least 22 so Ackley-20 has evaluations after its 21-point warmup" >&2
  exit 2
fi

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

run_group() {
  local benchmark="$1" context="$2" defaults="$3"
  local selected="${CONDITIONS:-$defaults}"
  local condition seed
  IFS=',' read -r -a CONDITION_ARR <<< "$selected"
  for condition in "${CONDITION_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      cmd=(
        bash scripts/run_plugins.sh
        "$benchmark" "$condition" "$seed"
        --budget "$BUDGET"
        --context "$context"
        --root "$ROOT"
      )
      [ "$LIST" = "1" ] && cmd+=(--list)
      [ "$FORCE" = "1" ] && cmd+=(--force)
      "${cmd[@]}"
    done
  done
}

case "$STAGE" in
  smoke)
    run_group hartmann6 blind "vanilla,cake,turbo,cake-turbo"
    run_group ackley20 blind "vanilla,cake,turbo,cake-turbo"
    run_group bolt_lora domain "vanilla,cake,turbo,cake-turbo,pibo,cake-turbo-pibo,llambo"
    ;;
  phase1)
    run_group hartmann6 blind "vanilla,cake,turbo,cake-turbo"
    run_group ackley10 blind "vanilla,cake,turbo,cake-turbo"
    run_group ackley20 blind "vanilla,cake,turbo,cake-turbo"
    ;;
  phase2)
    run_group bolt_lora domain "vanilla,cake,turbo,cake-turbo"
    ;;
  phase3)
    run_group bolt_lora domain "cake-turbo,cake-turbo-pibo"
    run_group bolt_lora generic "cake-turbo,cake-turbo-pibo"
    run_group bolt_lora misleading "cake-turbo,cake-turbo-pibo"
    ;;
  phase4)
    # LLAMBO proposes candidates directly, so CAKE's surrogate is not crossed
    # with it. This phase isolates the sampler slot against BoTorch.
    run_group ackley20 blind "vanilla,llambo"
    run_group bolt_lora domain "vanilla,llambo"
    ;;
esac
