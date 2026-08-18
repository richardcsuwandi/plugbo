#!/usr/bin/env bash
# BoLT LoRA mixed-type HPO (no textbook optimum). Always identity-revealed.
#
#   pip install -e '.[bolt]'   # once, for the Hugging Face emulator weights
#   ./scripts/run_bolt.sh
#   ./scripts/run_bolt.sh --backend vanilla,cake
#   ./scripts/run_bolt.sh --context generic --seed 42
#
# --backend   comma list, or all (vanilla,cake,sara-lenz,sara-lenz-cake,sara-only)
# --context    domain (default) | generic | misleading
# Completed and in-flight legs are skipped; pass --force to rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKEND="vanilla,cake,sara-lenz,sara-lenz-cake,sara-only"
CONTEXT="domain"
ROOT_OVERRIDE=""
FORCE=0
LIST=0
BUDGET=100
SEED=42
WARMUP=8
while [ $# -gt 0 ]; do
  case "$1" in
    --backend)
      BACKEND="${2:?}"
      shift 2
      ;;
    --context)
      CONTEXT="${2:?}"
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
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

case "$CONTEXT" in
  domain|generic|misleading) ;;
  *)
    echo "error: --context must be domain, generic, or misleading" >&2
    exit 1
    ;;
esac

source scripts/_compare_env.sh
export FORCE LIST BUDGET SEED WARMUP
CONTEXT_VARIANT="$CONTEXT"
export CONTEXT_VARIANT
source scripts/_run_lib.sh

BACKENDS="$(expand_backends "$BACKEND" "vanilla,cake,sara-lenz,sara-lenz-cake,sara-only")"
# shellcheck disable=SC2206
BACKEND_ARR=($BACKENDS)

if [ -n "$ROOT_OVERRIDE" ]; then
  ROOT="$ROOT_OVERRIDE"
elif [ "$CONTEXT" = "domain" ]; then
  ROOT="./results/logs/bolt_lora"
else
  ROOT="./results/logs/bolt_lora-${CONTEXT}"
fi

echo "benchmark=bolt_lora budget=$BUDGET seed=$SEED warmup=$WARMUP context=$CONTEXT backend=$BACKENDS provider=$PROVIDER model=$MODEL"
echo "root=$ROOT"
echo
mkdir -p "$ROOT"

for backend in "${BACKEND_ARR[@]}"; do
  run_leg "$ROOT/$backend" bolt_lora "$backend" revealed
  echo
done

plot_root "$ROOT" "BOLT LoRA HPO ($CONTEXT context, seed $SEED)"
echo
echo "Open $ROOT/compare.html  (or sara-viz --root $ROOT)"
