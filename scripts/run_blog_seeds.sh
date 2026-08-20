#!/usr/bin/env bash
# Extra seeds for the PlugBO blog showcase (not a 10-seed conference sweep).
#
# Two lanes, one LLM job each. Run them in two terminals so you stay at
# exactly two LLM-calling experiments:
#
#   ./scripts/run_blog_seeds_a.sh
#   ./scripts/run_blog_seeds_b.sh
#
# Lane A: hartmann6, gp_sample6, bolt misleading (RQ4 subset)
# Lane B: ackley10, bolt domain
# Free backends (vanilla/turbo/pibo) still go two-wide inside a lane.
# LLM backends go one-wide inside a lane.
#
#   ./scripts/run_blog_seeds.sh --lane a --list
#   ./scripts/run_blog_seeds.sh --status
#   ./scripts/run_blog_seeds.sh --lane a --seeds 43,44
#
# Completed seed×backend legs are skipped. Logs: /tmp/blog_seeds_logs/{a,b}/
set -euo pipefail
trap '' TTOU TTIN
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS="43,44"
SUITE="all"
LANE=""
BENCHMARKS=""
LIST=0
STATUS=0
FORCE=0
BUDGET=100
LOGDIR="${LOGDIR:-/tmp/blog_seeds_logs}"

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds|--seed) SEEDS="${2:?}"; shift 2 ;;
    --suite) SUITE="${2:?}"; shift 2 ;;
    --lane) LANE="${2:?}"; shift 2 ;;
    --benchmarks) BENCHMARKS="${2:?}"; shift 2 ;;
    --budget) BUDGET="${2:?}"; shift 2 ;;
    --logdir) LOGDIR="${2:?}"; shift 2 ;;
    --list) LIST=1; shift ;;
    --status) STATUS=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

case "$SUITE" in
  all|synthetics|bolt) ;;
  *)
    echo "error: --suite must be all, synthetics, or bolt" >&2
    exit 2
    ;;
esac

if [ "$STATUS" != "1" ]; then
  case "$LANE" in
    a|b) ;;
    *)
      echo "error: pick one lane so two terminals stay at 2 LLM jobs:" >&2
      echo "  ./scripts/run_blog_seeds_a.sh" >&2
      echo "  ./scripts/run_blog_seeds_b.sh" >&2
      echo "or: $0 --lane a|b" >&2
      exit 2
      ;;
  esac
fi

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

synth_root() {
  case "$1" in
    hartmann6) echo "./results/logs/hartmann6-compare" ;;
    ackley10) echo "./results/logs/ackley10-compare" ;;
    gp_sample6) echo "./results/logs/gp_sample6-blind" ;;
    *) echo "" ;;
  esac
}

bolt_root() {
  case "$1" in
    domain) echo "./results/logs/bolt_lora-compare" ;;
    misleading) echo "./results/logs/bolt_lora-misleading-compare" ;;
    *) echo "" ;;
  esac
}

lane_synth() {
  case "$LANE" in
    a) echo "hartmann6 gp_sample6" ;;
    b) echo "ackley10" ;;
  esac
}

lane_has_bolt_domain() {
  [ "$LANE" = b ]
}

lane_has_bolt_misleading() {
  [ "$LANE" = a ]
}

want_bench() {
  local name="$1"
  if [ -z "$BENCHMARKS" ]; then
    return 0
  fi
  case ",$BENCHMARKS," in
    *",$name,"*) return 0 ;;
    *) return 1 ;;
  esac
}

want_bolt_domain() {
  lane_has_bolt_domain || return 1
  [ -z "$BENCHMARKS" ] || want_bench bolt_lora || want_bench bolt_lora-domain
}

want_bolt_misleading() {
  lane_has_bolt_misleading || return 1
  [ -z "$BENCHMARKS" ] || want_bench bolt_lora || want_bench bolt_lora-misleading
}

if [ "$STATUS" = "1" ]; then
  python3 - "$SUITE" "$BENCHMARKS" <<'PY'
from pathlib import Path
import json, sys
suite = sys.argv[1]
wanted = [x for x in sys.argv[2].split(",") if x]
groups = []
if suite in ("all", "synthetics"):
    groups += [
        ("hartmann6  [lane a]", Path("results/logs/hartmann6-compare")),
        ("ackley10  [lane b]", Path("results/logs/ackley10-compare")),
        ("gp_sample6  [lane a]", Path("results/logs/gp_sample6-blind")),
    ]
if suite in ("all", "bolt"):
    groups += [
        ("bolt_lora/domain  [lane b]", Path("results/logs/bolt_lora-compare")),
        ("bolt_lora/misleading  [lane a]", Path("results/logs/bolt_lora-misleading-compare")),
    ]
if wanted:
    groups = [g for g in groups if any(w in g[0] for w in wanted)]
for label, root in groups:
    print(f"\n== {label} ==")
    if not root.is_dir():
        print("  missing")
        continue
    for cond in sorted(p for p in root.iterdir() if p.is_dir()):
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
            print(f"  {cond.name:16} none")
            continue
        bits = ", ".join(
            f"seed {seed}: {status} {n}/{budget}"
            for seed, status, n, budget in sorted(rows, key=lambda r: (r[0] is None, r[0] or 0))
        )
        print(f"  {cond.name:16} {bits}")
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

run_synth_backend() {
  local bench="$1" backend="$2" seed="$3"
  local root log
  root="$(synth_root "$bench")"
  log="$LOGDIR/${bench}_${backend}_seed${seed}.log"
  launch "$log" ./scripts/run_synthetic.sh "$bench" \
    --backend "$backend" --seed "$seed" --budget "$BUDGET" --root "$root" \
    "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
}

run_bolt_backend() {
  local context="$1" backend="$2" seed="$3"
  local root log
  root="$(bolt_root "$context")"
  log="$LOGDIR/bolt_${context}_${backend}_seed${seed}.log"
  if [ "$backend" = "pibo" ]; then
    launch "$log" ./scripts/run_plugins.sh bolt_lora pibo "$seed" \
      --budget "$BUDGET" --context "$context" --group "$root" \
      "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
    return 0
  fi
  launch "$log" ./scripts/run_bolt.sh \
    --context "$context" --backend "$backend" --seed "$seed" --budget "$BUDGET" \
    --root "$root" \
    "${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}" "${LIST_FLAG[@]+"${LIST_FLAG[@]}"}"
}

# One LLM-calling backend at a time. cake counts: it calls a kernel LLM.
run_llm_one() {
  "$@"
  wait_wave
}

summarize_root() {
  local root="$1"
  if [ "$LIST" = "1" ]; then
    echo "SUM  python3 -m benchmarks.summarize_compare --root $root"
    return 0
  fi
  python3 -m benchmarks.plot_compare --root "$root"
  python3 -m benchmarks.summarize_compare --root "$root"
}

echo "blog lane=$LANE seeds=$SEEDS suite=$SUITE budget=$BUDGET logdir=$LOGDIR"
echo "this lane runs 1 LLM job at a time; start the other lane in a second terminal"
echo

if [ "$SUITE" = "all" ] || [ "$SUITE" = "synthetics" ]; then
  for bench in $(lane_synth); do
    want_bench "$bench" || continue
    echo "== $bench (lane $LANE) =="
    for seed in "${SEED_ARR[@]}"; do
      echo "-- seed $seed free (vanilla+turbo, no campaign LLM) --"
      run_synth_backend "$bench" vanilla "$seed"
      run_synth_backend "$bench" turbo "$seed"
      wait_wave
      echo "-- seed $seed LLM (1 at a time) --"
      run_llm_one run_synth_backend "$bench" cake "$seed"
      run_llm_one run_synth_backend "$bench" sara-only "$seed"
      run_llm_one run_synth_backend "$bench" sara-lenz "$seed"
      run_llm_one run_synth_backend "$bench" sara-lenz-cake "$seed"
    done
    summarize_root "$(synth_root "$bench")"
    echo
  done
fi

if [ "$SUITE" = "all" ] || [ "$SUITE" = "bolt" ]; then
  if want_bolt_domain; then
    echo "== bolt_lora domain (lane $LANE) =="
    for seed in "${SEED_ARR[@]}"; do
      echo "-- seed $seed free (vanilla+turbo, then pibo) --"
      run_bolt_backend domain vanilla "$seed"
      run_bolt_backend domain turbo "$seed"
      wait_wave
      run_bolt_backend domain pibo "$seed"
      wait_wave
      echo "-- seed $seed LLM (1 at a time) --"
      run_llm_one run_bolt_backend domain cake "$seed"
      run_llm_one run_bolt_backend domain sara-only "$seed"
      run_llm_one run_bolt_backend domain sara-lenz "$seed"
      run_llm_one run_bolt_backend domain sara-lenz-cake "$seed"
    done
    summarize_root "$(bolt_root domain)"
    echo
  fi

  if want_bolt_misleading; then
    echo "== bolt_lora misleading RQ4 subset (lane $LANE) =="
    for seed in "${SEED_ARR[@]}"; do
      echo "-- seed $seed free (vanilla+turbo, then pibo) --"
      run_bolt_backend misleading vanilla "$seed"
      run_bolt_backend misleading turbo "$seed"
      wait_wave
      run_bolt_backend misleading pibo "$seed"
      wait_wave
      echo "-- seed $seed LLM (1 at a time) --"
      run_llm_one run_bolt_backend misleading sara-lenz "$seed"
      run_llm_one run_bolt_backend misleading sara-only "$seed"
    done
    summarize_root "$(bolt_root misleading)"
    echo
  fi
fi

if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "finished with ${#FAILS[@]} failed job(s): ${FAILS[*]}" >&2
  exit 1
fi
echo "done lane=$LANE. ./scripts/run_blog_seeds.sh --status"
echo "logs: $LOGDIR"
