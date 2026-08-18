#!/usr/bin/env bash
# Launches every not-yet-run experiment needed to answer this project's open
# research questions (see docs/observations.md for the full rationale behind
# each axis):
#
#   RQ1 (backend sweep, blind)    -- does the agentic loop help when there is
#                                    nothing to recall? lenz-only vs sara+lenz
#                                    vs sara+lenz+cake, identity hidden.
#   RQ2 (backend sweep, revealed) -- given the name, which backend actually
#                                    uses it? same 3-way ablation, identity
#                                    revealed.
#   RQ3 (disclosure sweep)        -- holding the backend fixed, how much of
#                                    the curve is identity? blind vs
#                                    revealed+shifted vs revealed+unshifted.
#   RQ4 (harness validity)        -- do RQ3's gaps collapse on gp_sample
#                                    (nothing to recall, nothing published)?
#                                    If they don't, the harness is leaking.
#   RQ5 (generalization)          -- do RQ1-3 findings hold on the paper's
#                                    other named benchmarks and the
#                                    constrained special case?
#
# RQ1 and RQ2 ARE the "lenz -> sara+lenz -> sara+lenz+cake" ablation for
# separating the LLM-agent effect from the kernel-evolution effect -- that
# already exists as run_benchmark_compare.sh / run_benchmark_noblind_compare.sh.
# Nothing new was added for it; this file just fills in the legs of it (and
# everything else) that haven't been run yet.
#
# Safety, checked by hand against `results/logs` on 2026-08-18:
#   - Every leg below writes to its OWN --root -- nothing here shares a
#     directory with anything already completed or already running.
#   - Two fixes upstream make this safe to run concurrently at all:
#     (1) the old unconditional `rm -rf "$ROOT"` in every compare script is
#         gone (scripts/run_*_compare.sh) -- reruns no longer delete anyone
#         else's in-flight data;
#     (2) run_noblind_compare.sh / run_benchmark_noblind_compare.sh now take
#         an explicit --root override, because their default root is keyed
#         only by benchmark name (not --config / --shift-only) -- running a
#         second config/shift variant against the same benchmark's DEFAULT
#         root would comingle two different backends' sandboxes in the same
#         folder. Every leg below that reuses a benchmark already present
#         elsewhere passes --root explicitly for exactly this reason.
#   - This script does NOT do automatic "already done" detection -- it is a
#     one-time batch, not idempotent. Re-running it later without re-checking
#     `results/logs` yourself could re-launch something already finished.
#   - Does NOT touch the independently-running `gp-disclosure` screen session
#     (gp_sample6, --config sara-lenz, launched earlier) -- that finishes on
#     its own; this script only adds gp_sample6's other two backends, under
#     their own --root.
#   - Waves run sequentially; legs WITHIN a wave run concurrently via
#     `screen` (immune to the tty-suspend issue from running plain `&`
#     background jobs in an interactive shell). Capped at ~2-3 concurrent
#     LLM-driven legs per wave -- ModelScope has been observed to drop
#     streaming connections under load (see llm/openai_client.py's retry
#     logic, added for exactly this failure mode).
#
# Scale/cost warning: this is 13 more 100-eval LLM-driven campaigns across 4
# waves. Expect multiple hours of wall-clock time and real API spend. Run one
# wave at a time if you'd rather review results before committing to the rest.
#
# Usage:
#   ./scripts/run_pending_experiments.sh          # all waves, sequentially
#   ./scripts/run_pending_experiments.sh 2        # only wave 2
#   ./scripts/run_pending_experiments.sh list     # print the plan, run nothing
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_compare_env.sh

LOGDIR=/tmp/agentic-bo-logs
mkdir -p "$LOGDIR"

launch() {
  # $1 = screen session name, $2 = log filename, $3.. = command + args
  local name="$1" log="$2"
  shift 2
  local quoted
  quoted=$(printf '%q ' "$@")
  screen -dmS "$name" bash -c "cd $(printf '%q' "$(pwd)") && $quoted > $(printf '%q' "$LOGDIR/$log") 2>&1"
  echo "  launched: $name  (log: $LOGDIR/$log)"
}

wait_for_wave() {
  echo "  waiting for: $*"
  while true; do
    local remaining=0
    for n in "$@"; do
      screen -ls 2>/dev/null | grep -q "\.$n[[:space:]]" && remaining=1
    done
    [ "$remaining" = "0" ] && break
    sleep 30
  done
  echo "  wave complete."
}

wave1() {
  echo "=== Wave 1: fill existing partial 3-way sweeps (RQ1, RQ2) ==="
  echo "  running vanilla/blind inline (no LLM, instant)"
  python3 -m benchmarks.run_blind_baseline --benchmark hartmann6 --budget 100 --seed 42 \
    --root results/logs/hartmann6-compare/vanilla --policy vanilla

  launch w1-blind-cake blind-cake.log \
    python3 -m benchmarks.run_blind_test --benchmark hartmann6 --provider "$PROVIDER" --model "$MODEL" \
    --base-url "$BASE_URL" --api-key "$API_KEY" --budget 100 --seed 42 \
    --root results/logs/hartmann6-compare/sara-lenz-cake --surrogate cake --acqf noisy_logei \
    --extra-body "$EXTRA_BODY" --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL" \
    --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" \
    --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"

  launch w1-revealed-cake revealed-cake.log \
    python3 -m benchmarks.run_noblind_test --benchmark hartmann6 --provider "$PROVIDER" --model "$MODEL" \
    --base-url "$BASE_URL" --api-key "$API_KEY" --budget 100 --seed 42 \
    --root results/logs/hartmann6-noblind-compare-3config/sara-lenz-cake --surrogate cake --acqf noisy_logei \
    --extra-body "$EXTRA_BODY" --one-shot-tol 0.01 --kernel-llm-provider "$KERNEL_LLM_PROVIDER" \
    --kernel-llm-model "$KERNEL_LLM_MODEL" --kernel-llm-base-url "$KERNEL_LLM_BASE_URL" \
    --kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV" --kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY"

  launch w1-disclosure-vanilla disclosure-vanilla.log \
    ./scripts/run_noblind_compare.sh hartmann6 100 42 --config vanilla \
    --root ./results/logs/hartmann6-noblind-compare-vanilla

  wait_for_wave w1-blind-cake w1-revealed-cake w1-disclosure-vanilla
}

wave2() {
  echo "=== Wave 2: remaining disclosure-sweep backend + shift-only backend sweep (RQ3) ==="
  launch w2-disclosure-cake disclosure-cake.log \
    ./scripts/run_noblind_compare.sh hartmann6 100 42 --config sara-lenz-cake \
    --root ./results/logs/hartmann6-noblind-compare-cake

  launch w2-shift-only shift-only.log \
    ./scripts/run_benchmark_noblind_compare.sh hartmann6 100 42 --shift-only \
    --root ./results/logs/hartmann6-noblind-compare-3config-shifted

  wait_for_wave w2-disclosure-cake w2-shift-only
}

wave3() {
  echo "=== Wave 3: generalize to the paper's other named benchmarks (RQ5) ==="
  launch w3-branin branin.log ./scripts/run_noblind_compare.sh branin 100 42
  launch w3-ackley10 ackley10.log ./scripts/run_noblind_compare.sh ackley10 100 42
  launch w3-ackley20 ackley20.log ./scripts/run_noblind_compare.sh ackley20 100 42

  wait_for_wave w3-branin w3-ackley10 w3-ackley20
}

wave4() {
  echo "=== Wave 4: constrained special case + remaining gp_sample negative-control backends (RQ4, RQ5) ==="
  launch w4-constrained constrained.log ./scripts/run_noblind_compare.sh constrained_hartmann6 100 42

  launch w4-gp-vanilla gp-vanilla.log \
    ./scripts/run_noblind_compare.sh gp_sample6 100 42 --config vanilla \
    --root ./results/logs/gp_sample6-noblind-compare-vanilla

  launch w4-gp-cake gp-cake.log \
    ./scripts/run_noblind_compare.sh gp_sample6 100 42 --config sara-lenz-cake \
    --root ./results/logs/gp_sample6-noblind-compare-cake

  wait_for_wave w4-constrained w4-gp-vanilla w4-gp-cake
}

list_plan() {
  cat <<'EOF'
Wave 1 (RQ1/RQ2 -- fill existing partial sweeps):
  - hartmann6-compare/vanilla                       (inline, no LLM)
  - hartmann6-compare/sara-lenz-cake                (screen: w1-blind-cake)
  - hartmann6-noblind-compare-3config/sara-lenz-cake (screen: w1-revealed-cake)
  - hartmann6-noblind-compare-vanilla/{blind,noblind-shift,noblind} (screen: w1-disclosure-vanilla)

Wave 2 (RQ3 -- remaining disclosure backend + shift-only backend sweep):
  - hartmann6-noblind-compare-cake/{blind,noblind-shift,noblind}    (screen: w2-disclosure-cake)
  - hartmann6-noblind-compare-3config-shifted/{vanilla,sara-lenz,sara-lenz-cake} (screen: w2-shift-only)

Wave 3 (RQ5 -- generalize to paper's other benchmarks, disclosure sweep):
  - branin-noblind-compare/{blind,noblind-shift,noblind}   (screen: w3-branin)
  - ackley10-noblind-compare/{...}                          (screen: w3-ackley10)
  - ackley20-noblind-compare/{...}                          (screen: w3-ackley20)

Wave 4 (RQ4/RQ5 -- constrained special case + gp_sample negative control):
  - constrained_hartmann6-noblind-compare/{...}             (screen: w4-constrained)
  - gp_sample6-noblind-compare-vanilla/{...}  (no LLM)       (screen: w4-gp-vanilla)
  - gp_sample6-noblind-compare-cake/{...}                    (screen: w4-gp-cake)
EOF
}

ARG="${1:-all}"
case "$ARG" in
  1) wave1 ;;
  2) wave2 ;;
  3) wave3 ;;
  4) wave4 ;;
  all) wave1; wave2; wave3; wave4 ;;
  list) list_plan; exit 0 ;;
  *) echo "usage: $0 [1|2|3|4|all|list]" >&2; exit 1 ;;
esac

echo
echo "Requested wave(s) complete. Regenerate every chart with:"
echo "  python3 -m benchmarks.plot_all"
