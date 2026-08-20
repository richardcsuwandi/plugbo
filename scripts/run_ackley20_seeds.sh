#!/usr/bin/env bash
# Extra seeds for ackley20 -- brings it up to the same 3-seed (42, 43, 44)
# coverage as every other synthetic benchmark, so it clears the merged
# viewer's "2+ seeds" bar (results/logs/ackley20-compare currently has only
# seed 42 for every condition, including sara-lenz-turbo).
#
#   ./scripts/run_ackley20_seeds.sh
#   ./scripts/run_ackley20_seeds.sh --list
#   ./scripts/run_ackley20_seeds.sh --status
#
# Free backends (vanilla, turbo) run 2-wide. LLM backends (cake, sara-only,
# sara-lenz, sara-lenz-cake, sara-lenz-turbo) run one at a time, matching the
# concurrency discipline of run_blog_seeds.sh / run_plugin_extras.sh. Meant
# to run in its own terminal alongside ./scripts/run_bolt_generic_seeds.sh
# (independent benchmark groups, safe to run in parallel).
#
# Completed seed x backend legs are skipped; pass --force to rerun.
set -euo pipefail
trap '' TTOU TTIN
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS="43,44"
BUDGET=100
ROOT="./results/logs/ackley20-compare"
LOGDIR="${LOGDIR:-/tmp/blog_seeds_logs}"
LIST=0
STATUS=0
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds|--seed) SEEDS="${2:?}"; shift 2 ;;
    --budget) BUDGET="${2:?}"; shift 2 ;;
    --root) ROOT="${2:?}"; shift 2 ;;
    --logdir) LOGDIR="${2:?}"; shift 2 ;;
    --list) LIST=1; shift ;;
    --status) STATUS=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [ "$STATUS" = "1" ]; then
  python3 - "$ROOT" <<'PY'
from pathlib import Path
import json, sys

root = Path(sys.argv[1])
if not root.is_dir():
    print("missing")
    raise SystemExit
for cond in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_answers"):
    rows = []
    for meta_path in cond.glob("sandbox_*/run_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        state_path = meta_path.parent / "state.json"
        n = 0
        if state_path.is_file():
            trials = json.loads(state_path.read_text()).get("trials") or []
            n = sum(1 for t in trials if t.get("status") == "observed")
        rows.append((meta.get("seed"), meta.get("status") or "?", n, meta.get("budget")))
    bits = ", ".join(
        f"seed {seed}: {status} {n}/{budget}"
        for seed, status, n, budget in sorted(rows, key=lambda r: (r[0] is None, r[0] or 0))
    )
    print(f"{cond.name:16} {bits}")
PY
  exit 0
fi

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1
FORCE_FLAG=()
LIST_FLAG=()
[ "$FORCE" = "1" ] && FORCE_FLAG=(--force)
[ "$LIST" = "1" ] && LIST_FLAG=(--list)

PIDS=()
FAILS=()

launch() {
  local log="$1"
  shift
  echo "CMD  $*  > $log"
  if [ "$LIST" = "1" ]; then
    return 0
  fi
  nohup "$@" >"$log" 2>&1 </dev/null &
  PIDS+=("$!")
}

wait_wave() {
  local pid st
  if [ "$LIST" = "1" ]; then
    PIDS=()
    return 0
  fi
  for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
    if wait "$pid"; then
      echo "OK   pid=$pid"
    else
      st=$?
      echo "FAIL pid=$pid exit=$st"
      FAILS+=("$pid:$st")
    fi
  done
  PIDS=()
}

run_backend() {
  local backend="$1" seed="$2"
  local log="$LOGDIR/ackley20_${backend}_seed${seed}.log"
  launch "$log" ./scripts/run_synthetic.sh ackley20 \
    --backend "$backend" --seed "$seed" --budget "$BUDGET" --root "$ROOT" \
    "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
}

echo "ackley20 seeds=$SEEDS budget=$BUDGET root=$ROOT logdir=$LOGDIR"
echo "one LLM job at a time"
echo

for seed in "${SEED_ARR[@]}"; do
  echo "== seed $seed free (vanilla+turbo, no campaign LLM) =="
  run_backend vanilla "$seed"
  run_backend turbo "$seed"
  wait_wave

  echo "== seed $seed LLM (1 at a time) =="
  run_backend cake "$seed"; wait_wave
  run_backend sara-only "$seed"; wait_wave
  run_backend sara-lenz "$seed"; wait_wave
  run_backend sara-lenz-cake "$seed"; wait_wave
  run_backend sara-lenz-turbo "$seed"; wait_wave
done

if [ "$LIST" != "1" ]; then
  python3 -m benchmarks.plot_compare --root "$ROOT"
  python3 -m benchmarks.summarize_compare --root "$ROOT"
fi

if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "finished with ${#FAILS[@]} failed job(s): ${FAILS[*]}" >&2
  exit 1
fi
echo "done. ./scripts/run_ackley20_seeds.sh --status"
echo "logs: $LOGDIR"
