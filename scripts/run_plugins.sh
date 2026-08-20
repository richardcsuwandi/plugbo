#!/usr/bin/env bash
# Run one fixed PlugBO plugin condition for one benchmark and seed.
# Local experiment driver: intentionally not part of the committed harness.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -lt 3 ]; then
  echo "usage: $0 <benchmark> <condition> <seed> [--budget N] [--context blind|domain|generic|misleading] [--root DIR | --group DIR] [--list] [--force]" >&2
  exit 2
fi

BENCHMARK="$1"
CONDITION="$2"
SEED="$3"
shift 3

BUDGET=100
CONTEXT="blind"
ROOT="./results/logs/plugin-matrix"
GROUP_OVERRIDE=""
LIST=0
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --budget) BUDGET="${2:?}"; shift 2 ;;
    --context) CONTEXT="${2:?}"; shift 2 ;;
    --root) ROOT="${2:?}"; shift 2 ;;
    --group) GROUP_OVERRIDE="${2:?}"; shift 2 ;;
    --list) LIST=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

case "$CONTEXT" in
  blind|domain|generic|misleading) ;;
  *) echo "error: invalid context '$CONTEXT'" >&2; exit 2 ;;
esac

source scripts/_compare_env.sh
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

GROUP="${GROUP_OVERRIDE:-$ROOT/${BENCHMARK}-${CONTEXT}}"
OUT="$GROUP/$CONDITION"

existing_status="$("$PYTHON" - "$OUT" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
matches = []
for path in root.glob("sandbox_*/run_meta.json"):
    try:
        meta = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if meta.get("seed") == seed:
        matches.append((path.stat().st_mtime, meta.get("status") or "unknown"))
print(max(matches)[1] if matches else "missing")
PY
)"

if [ "$FORCE" != "1" ] && { [ "$existing_status" = "completed" ] || [ "$existing_status" = "running" ]; }; then
  echo "SKIP benchmark=$BENCHMARK condition=$CONDITION seed=$SEED status=$existing_status"
  exit 0
fi

args=(
  -m benchmarks.run_blind_baseline
  --benchmark "$BENCHMARK"
  --policy vanilla
  --budget "$BUDGET"
  --seed "$SEED"
  --root "$OUT"
)

if [ "$BENCHMARK" = "bolt_lora" ]; then
  if [ "$CONTEXT" = "blind" ]; then
    echo "error: BoLT plugin studies require domain, generic, or misleading context" >&2
    exit 2
  fi
  args+=(--reveal --context-variant "$CONTEXT" --warmup 8)
elif [ "$CONTEXT" != "blind" ]; then
  echo "error: non-BoLT plugin studies currently use context=blind" >&2
  exit 2
fi

needs_cake=0
needs_llambo=0
case "$CONDITION" in
  vanilla) ;;
  cake) args+=(--surrogate cake); needs_cake=1 ;;
  turbo) args+=(--region turbo) ;;
  cake-turbo) args+=(--surrogate cake --region turbo); needs_cake=1 ;;
  pibo) args+=(--prior-fixture "bolt-$CONTEXT") ;;
  cake-turbo-pibo)
    args+=(--surrogate cake --region turbo --prior-fixture "bolt-$CONTEXT")
    needs_cake=1
    ;;
  llambo) args+=(--sampler llambo); needs_llambo=1 ;;
  cake-turbo-llambo)
    args+=(--surrogate cake --region turbo --sampler llambo)
    needs_cake=1
    needs_llambo=1
    ;;
  *) echo "error: unknown condition '$CONDITION'" >&2; exit 2 ;;
esac

if [[ "$CONDITION" == *pibo* ]] && [ "$BENCHMARK" != "bolt_lora" ]; then
  echo "error: deterministic πBO fixtures are defined only for bolt_lora" >&2
  exit 2
fi

if [ "$needs_cake" = "1" ] || [ "$needs_llambo" = "1" ]; then
  args+=(--llm-provider "$PROVIDER" --llm-model "$MODEL")
  [ -n "$BASE_URL" ] && args+=(--llm-base-url "$BASE_URL")
  [ -n "$EXTRA_BODY" ] && args+=(--llm-extra-body "$EXTRA_BODY")
  [ -n "${KERNEL_LLM_API_KEY_ENV:-}" ] && args+=(--llm-api-key-env "$KERNEL_LLM_API_KEY_ENV")
fi
if [ "$needs_cake" = "1" ]; then
  args+=(--kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL")
  [ -n "$KERNEL_LLM_BASE_URL" ] && args+=(--kernel-llm-base-url "$KERNEL_LLM_BASE_URL")
  [ -n "$KERNEL_LLM_API_KEY_ENV" ] && args+=(--kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV")
  [ -n "$KERNEL_LLM_EXTRA_BODY" ] && args+=(--kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY")
fi

echo "RUN benchmark=$BENCHMARK context=$CONTEXT condition=$CONDITION seed=$SEED budget=$BUDGET"
printf 'CMD'
printf ' %q' "$PYTHON" "${args[@]}"
printf '\n'
if [ "$LIST" = "1" ]; then
  exit 0
fi

mkdir -p "$OUT"
"$PYTHON" "${args[@]}"
"$PYTHON" -m benchmarks.plot_compare --root "$GROUP" --title "$BENCHMARK $CONTEXT plugin comparison"
"$PYTHON" -m benchmarks.summarize_compare --root "$GROUP"
