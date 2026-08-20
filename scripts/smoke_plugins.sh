#!/usr/bin/env bash
# Smoke-test PlugBO plugins before a v1 push or a real experiment sweep.
#
# Default (no API): unit tests + CLI wiring + short Branin loops
# (vanilla, TuRBO, πBO). CAKE/LLAMBO are checked for inheritance/wiring only.
#
#   ./scripts/smoke_plugins.sh
#   ./scripts/smoke_plugins.sh --live          # one CAKE evolve + one LLAMBO sample
#   ./scripts/smoke_plugins.sh --baseline      # also hartmann6 vanilla vs turbo, budget 20
#
# After this passes, v1 is safe to push. Next experiments (not this script):
#   python3 -m benchmarks.run_blind_baseline --benchmark hartmann6 --budget 30 --seed 42 --policy vanilla
#   python3 -m benchmarks.run_blind_baseline --benchmark hartmann6 --budget 30 --seed 42 --policy turbo
#   python3 -m benchmarks.run_blind_baseline --benchmark hartmann6 --budget 30 --seed 42 --policy cake
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LIVE=0
BASELINE=0
PYTEST=1
for arg in "$@"; do
  case "$arg" in
    --live) LIVE=1 ;;
    --baseline) BASELINE=1 ;;
    --no-pytest) PYTEST=0 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg (expected --live, --baseline, --no-pytest)" >&2
      exit 2
      ;;
  esac
done

PYTHON="${PYTHON:-./.venv/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

if [ "$PYTEST" -eq 1 ]; then
  echo "=== unit tests (plugins + cake, not slow) ==="
  "$PYTHON" -m pytest tests/test_lenz_plugins.py tests/test_lenz_cake.py -q -m "not slow"
  echo
fi

SMOKE_ARGS=()
if [ "$LIVE" -eq 1 ]; then
  # shellcheck source=scripts/_compare_env.sh
  source scripts/_compare_env.sh
  export PROVIDER MODEL BASE_URL API_KEY EXTRA_BODY KERNEL_LLM_API_KEY_ENV
  if [ -n "${API_KEY:-}" ] && [ -n "${PROVIDER:-}" ]; then
    case "$PROVIDER" in
      anthropic) export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$API_KEY}" ;;
      openai|openai-compatible) export OPENAI_API_KEY="${OPENAI_API_KEY:-$API_KEY}" ;;
    esac
  fi
  echo "live LLM: provider=$PROVIDER model=$MODEL"
  SMOKE_ARGS+=(--live)
fi

echo "=== plugin capabilities ==="
"$PYTHON" scripts/smoke_plugins.py "${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}"

if [ "$BASELINE" -eq 1 ]; then
  echo
  echo "=== no-agent hartmann6 (budget 20, seed 42) ==="
  ROOT="./results/logs/plugin-smoke"
  mkdir -p "$ROOT"
  for policy in vanilla turbo; do
    echo
    echo "--- policy $policy ---"
    "$PYTHON" -m benchmarks.run_blind_baseline \
      --benchmark hartmann6 --budget 20 --seed 42 \
      --root "$ROOT/$policy" --policy "$policy"
  done
  if [ "$LIVE" -eq 1 ]; then
    echo
    echo "--- policy cake (live kernel LLM) ---"
    "$PYTHON" -m benchmarks.run_blind_baseline \
      --benchmark hartmann6 --budget 20 --seed 42 \
      --root "$ROOT/cake" --policy cake \
      --llm-provider "$PROVIDER" --llm-model "$MODEL" \
      ${BASE_URL:+--llm-base-url "$BASE_URL"} \
      ${KERNEL_LLM_API_KEY_ENV:+--llm-api-key-env "$KERNEL_LLM_API_KEY_ENV"} \
      ${EXTRA_BODY:+--llm-extra-body "$EXTRA_BODY"}
  fi
  echo
  echo "baseline sandboxes under $ROOT/"
fi
