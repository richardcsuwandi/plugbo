# Sourced (not executed) by the run_*_compare.sh scripts -- resolves
# PROVIDER/MODEL/BASE_URL/API_KEY/EXTRA_BODY/KERNEL_LLM_* from env vars and
# .env, with the same ModelScope-by-default behavior across all of them, so
# that behavior only needs to be defined (and fixed, if it ever needs fixing)
# once. Assumes the caller has already `cd`ed to the repo root and set -u.

if [ -f .env ]; then
  set -a; source .env; set +a
fi

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

# True iff $1 is a benchmark name known to be constrained -- used to
# auto-skip the cake condition (cake doesn't support constrained studies)
# without paying gp_sample's full multi-start f_opt search just to check.
is_constrained_benchmark() {
  [ "$1" = "constrained_hartmann6" ]
}
