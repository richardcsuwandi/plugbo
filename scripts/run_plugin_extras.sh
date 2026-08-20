#!/usr/bin/env bash
# Follow-up plugin-slot experiments for the blog post, using the agent
# (Sara+lenz) rather than the scripted-only conditions run_plugins.sh
# already covers:
#
#   sara-lenz-turbo  region slot pinned to TuRBO under the agent
#   sara-lenz-pibo   prior slot pinned to a scripted pi-BO belief under the agent
#
# Meant to run AFTER ./scripts/run_blog_seeds.sh --lane a/b finish, so it
# doesn't add a third concurrent LLM-calling process on top of the two-lane
# budget those scripts are deliberately capped at.
#
#   ./scripts/run_plugin_extras.sh
#   ./scripts/run_plugin_extras.sh --list
#   ./scripts/run_plugin_extras.sh --status
#
# ackley20 sara-lenz-turbo: seed 42 only, budget 100 -- matches every other
# condition already in results/logs/ackley20-compare/ (no 43/44 data exists
# there to match; this overlays directly onto the existing compare.html).
# ackley20 is the paper's highest-dimensional synthetic benchmark, so it's
# the sharpest test of whether a trust region helps an agent-driven search
# the way it helps the scripted `turbo` baseline -- and, per the blog's
# Hartmann6-vs-Ackley10 discussion, whether the instability seen from
# pinning CAKE under an agent is CAKE-specific or shows up under any pinned
# plugin.
#
# bolt_lora sara-lenz-pibo: domain + misleading contexts, seeds 42/43/44,
# budget 100, warmup 8 -- matches the seeds/budget/warmup every other
# bolt_lora condition uses (including the seed 43/44 legs the blog-seeds
# sweep adds), reusing the same benchmarks/priors.py belief fixtures the
# scripted `pibo` baseline already uses, so it lands in the same
# compare.html and overlays directly.
#
# Completed and in-flight legs are skipped; pass --force to rerun.
set -euo pipefail
trap '' TTOU TTIN
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUDGET=100
LOGDIR="${LOGDIR:-/tmp/blog_seeds_logs}"
LIST=0
STATUS=0
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --budget) BUDGET="${2:?}"; shift 2 ;;
    --logdir) LOGDIR="${2:?}"; shift 2 ;;
    --list) LIST=1; shift ;;
    --status) STATUS=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,29p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

if [ "$STATUS" = "1" ]; then
  python3 - <<'PY'
from pathlib import Path
import json

groups = [
    ("ackley20 sara-lenz-turbo", Path("results/logs/ackley20-compare/sara-lenz-turbo")),
    ("bolt_lora/domain sara-lenz-pibo", Path("results/logs/bolt_lora-compare/sara-lenz-pibo")),
    ("bolt_lora/misleading sara-lenz-pibo", Path("results/logs/bolt_lora-misleading-compare/sara-lenz-pibo")),
]
for label, cond in groups:
    print(f"\n== {label} ==")
    if not cond.is_dir():
        print("  missing")
        continue
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
    if not rows:
        print("  none")
        continue
    for seed, status, n, budget in sorted(rows, key=lambda r: (r[0] is None, r[0] or 0)):
        print(f"  seed {seed}: {status} {n}/{budget}")
PY
  exit 0
fi

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

echo "plugin-extras budget=$BUDGET logdir=$LOGDIR"
echo "one LLM job at a time, same as the main blog-seeds sweep"
echo

echo "== ackley20 sara-lenz-turbo (seed 42) =="
launch "$LOGDIR/ackley20_sara-lenz-turbo_seed42.log" \
  ./scripts/run_synthetic.sh ackley20 --backend sara-lenz-turbo --seed 42 \
  --budget "$BUDGET" --root ./results/logs/ackley20-compare \
  "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
wait_wave
echo

echo "== bolt_lora sara-lenz-pibo (domain + misleading, seeds 42/43/44) =="
for seed in 42 43 44; do
  echo "-- seed $seed domain --"
  launch "$LOGDIR/bolt_domain_sara-lenz-pibo_seed${seed}.log" \
    ./scripts/run_bolt.sh --context domain --backend sara-lenz-pibo --seed "$seed" \
    --budget "$BUDGET" --root ./results/logs/bolt_lora-compare \
    "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
  wait_wave

  echo "-- seed $seed misleading --"
  launch "$LOGDIR/bolt_misleading_sara-lenz-pibo_seed${seed}.log" \
    ./scripts/run_bolt.sh --context misleading --backend sara-lenz-pibo --seed "$seed" \
    --budget "$BUDGET" --root ./results/logs/bolt_lora-misleading-compare \
    "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
  wait_wave
done
echo

if [ "$LIST" != "1" ]; then
  python3 -m benchmarks.plot_compare --root ./results/logs/ackley20-compare
  python3 -m benchmarks.summarize_compare --root ./results/logs/ackley20-compare
  python3 -m benchmarks.plot_compare --root ./results/logs/bolt_lora-compare
  python3 -m benchmarks.summarize_compare --root ./results/logs/bolt_lora-compare
  python3 -m benchmarks.plot_compare --root ./results/logs/bolt_lora-misleading-compare
  python3 -m benchmarks.summarize_compare --root ./results/logs/bolt_lora-misleading-compare
fi

if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "finished with ${#FAILS[@]} failed job(s): ${FAILS[*]}" >&2
  exit 1
fi
echo "done. ./scripts/run_plugin_extras.sh --status"
echo "logs: $LOGDIR"
