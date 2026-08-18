#!/usr/bin/env bash
# Runs the blind hartmann6 benchmark under three conditions -- sara+lenz
# (cake off), sara+lenz+cake, vanilla BO (no sara, no LLM) -- in that order
# (LLM-driven conditions first so you see whether the LLM is working before
# waiting on the instant no-LLM baseline), all pinned to the same acquisition
# function and the same seed (so all three see the identical hidden
# renaming/shift transform AND the same Sobol warm-start; after that the
# methods diverge), then overlays the resulting regret curves into
# one HTML page.
#
# Usage:
#   ./scripts/run_hartmann6_compare.sh [budget] [seed]
#
# Defaults to ModelScope (openai-compatible endpoint) driving Qwen models --
# reads MODELSCOPE_API_KEY / MODELSCOPE_BASE_URL from .env at repo root.
# Override provider/model/creds via env vars, e.g.:
#   PROVIDER=anthropic MODEL=claude-opus-5 ./scripts/run_hartmann6_compare.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

BUDGET="${1:-50}"
SEED="${2:-42}"
ROOT="./results/logs/hartmann6-compare"
ACQF="${ACQF:-noisy_logei}"

PROVIDER="${PROVIDER:-openai-compatible}"
MODEL="${MODEL:-Qwen-Ambassador/Qwen3.8-Max}"

# MODELSCOPE_* creds only apply when we're actually pointed at ModelScope --
# if PROVIDER is overridden (e.g. anthropic/openai), leave BASE_URL/API_KEY
# empty by default so each client falls back to its own standard env var
# (ANTHROPIC_API_KEY / OPENAI_API_KEY) instead of silently using ModelScope's.
if [ "$PROVIDER" = "openai-compatible" ]; then
  BASE_URL="${BASE_URL:-${MODELSCOPE_BASE_URL:-}}"
  API_KEY="${API_KEY:-${MODELSCOPE_API_KEY:-}}"
else
  BASE_URL="${BASE_URL:-}"
  API_KEY="${API_KEY:-}"
fi

KERNEL_LLM_PROVIDER="${KERNEL_LLM_PROVIDER:-$PROVIDER}"
KERNEL_LLM_MODEL="${KERNEL_LLM_MODEL:-$MODEL}"
KERNEL_LLM_BASE_URL="${KERNEL_LLM_BASE_URL:-$BASE_URL}"
KERNEL_LLM_API_KEY_ENV="${KERNEL_LLM_API_KEY_ENV:-MODELSCOPE_API_KEY}"

# Qwen3/DashScope-style models stream a hidden "thinking" pass before the visible
# content, which is most of what makes each call slow -- disabled by default when
# talking to ModelScope; override with EXTRA_BODY='{}' to leave thinking on, or
# EXTRA_BODY=... to pass something else through.
if [ -z "${EXTRA_BODY+x}" ] && [ "$PROVIDER" = "openai-compatible" ]; then
  EXTRA_BODY='{"enable_thinking": false}'
fi
EXTRA_BODY="${EXTRA_BODY:-}"
KERNEL_LLM_EXTRA_BODY="${KERNEL_LLM_EXTRA_BODY:-$EXTRA_BODY}"

if [ "$PROVIDER" = "openai-compatible" ] && [ -z "$BASE_URL" ]; then
  echo "error: --base-url required for provider 'openai-compatible' -- set MODELSCOPE_BASE_URL in .env, or pass BASE_URL=..." >&2
  exit 1
fi

echo "budget=$BUDGET seed=$SEED acqf=$ACQF provider=$PROVIDER model=$MODEL base_url=${BASE_URL:-<default>}"
echo

# Never rm -rf here: $ROOT is shared across invocations (same fixed path
# every time), so wiping it deletes any other run still writing into it
# concurrently. plot_compare.py already picks the latest sandbox_<token>/ per
# condition by mtime, so leftover sandboxes from old runs no longer leak into
# compare.html -- they just accumulate harmlessly.
mkdir -p "$ROOT"

echo "=== [1/3] sara + lenz (cake off) ==="
python3 -m benchmarks.run_blind_test \
  --benchmark hartmann6 --provider "$PROVIDER" --model "$MODEL" \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-lenz" \
  --surrogate fixed --acqf "$ACQF" \
  --extra-body "$EXTRA_BODY"

echo
echo "=== [2/3] sara + lenz + cake ==="
python3 -m benchmarks.run_blind_test \
  --benchmark hartmann6 --provider "$PROVIDER" --model "$MODEL" \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --budget "$BUDGET" --seed "$SEED" --root "$ROOT/sara-lenz-cake" \
  --surrogate cake --acqf "$ACQF" \
  --extra-body "$EXTRA_BODY" \
  --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
  --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
  --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"

echo
echo "=== [3/3] vanilla BO (no sara, no LLM) ==="
python3 -m benchmarks.run_blind_baseline \
  --benchmark hartmann6 --budget "$BUDGET" --seed "$SEED" \
  --root "$ROOT/vanilla" --policy vanilla

echo
echo "=== plotting comparison ==="
python3 -m benchmarks.plot_compare --root "$ROOT"

echo
echo "Open $ROOT/compare.html for the overlaid regret chart."
echo "Or browse each run's own trace/state: sara-viz --root $ROOT"
