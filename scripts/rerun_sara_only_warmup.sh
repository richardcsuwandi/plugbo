#!/usr/bin/env bash
# Sequential sara-only reruns for the six compares whose first evaluation
# used warmup 0 instead of the shared Sobol design. One ModelScope key,
# so these must not run in parallel.
#
#   ./scripts/rerun_sara_only_warmup.sh
#   ./scripts/rerun_sara_only_warmup.sh --list
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LIST=0
if [ "${1:-}" = "--list" ]; then
  LIST=1
elif [ $# -gt 0 ]; then
  echo "usage: $0 [--list]" >&2
  exit 2
fi

run() {
  local label="$1"
  shift
  echo
  echo "======== $label ========"
  if [ "$LIST" = "1" ]; then
    echo "LIST  $*"
    return 0
  fi
  "$@"
}

failed=()
run_or_record() {
  local label="$1"
  shift
  if ! run "$label" "$@"; then
    echo "FAIL  $label" >&2
    failed+=("$label")
  fi
}

# Keep going after a failed leg so one 429 does not skip the rest.
set +e
run_or_record hartmann6-compare \
  ./scripts/rerun_sara_only.sh hartmann6
run_or_record ackley10-compare \
  ./scripts/rerun_sara_only.sh ackley10
run_or_record ackley20-compare \
  ./scripts/rerun_sara_only.sh ackley20
run_or_record bolt_lora-compare \
  ./scripts/run_bolt.sh --backend sara-only --force \
  --root ./results/logs/bolt_lora-compare
run_or_record bolt_lora-generic-compare \
  ./scripts/run_bolt.sh --backend sara-only --context generic --force \
  --root ./results/logs/bolt_lora-generic-compare
run_or_record bolt_lora-misleading-compare \
  ./scripts/run_bolt.sh --backend sara-only --context misleading --force \
  --root ./results/logs/bolt_lora-misleading-compare
set -e

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All six sara-only legs finished."
else
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
