# Sourced by run_synthetic.sh and run_bolt.sh after _compare_env.sh.
# Shared: skip-if-done, LLM flags, one backend × disclosure leg.

# Line-buffer Python so `> log 2>&1 &` shows eval progress immediately.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

FORCE="${FORCE:-0}"
LIST="${LIST:-0}"
BUDGET="${BUDGET:-100}"
SEED="${SEED:-42}"
ACQF="${ACQF:-noisy_logei}"
WARMUP="${WARMUP:-}"
CONTEXT_VARIANT="${CONTEXT_VARIANT:-domain}"

_kernel_override() {
  [ "$KERNEL_LLM_PROVIDER" != "$PROVIDER" ] || [ "$KERNEL_LLM_MODEL" != "$MODEL" ] \
    || [ "${KERNEL_LLM_BASE_URL:-}" != "${BASE_URL:-}" ]
}

skip_or_run() {
  local cond_dir="$1"
  local label="$2"
  shift 2
  local st
  st=$(condition_status "$cond_dir" "${SEED:-}")
  if [ "$FORCE" != "1" ]; then
    case "$st" in
      completed|running)
        echo "SKIP  $label  seed=${SEED:-?} ($st)"
        return 0
        ;;
    esac
  fi
  echo "RUN   $label  seed=${SEED:-?} (status=$st)"
  if [ "$LIST" = "1" ]; then
    return 0
  fi
  mkdir -p "$cond_dir"
  "$@"
}

# One condition directory: $1=out_dir $2=benchmark $3=backend $4=disclosure
# disclosure: blind | revealed | revealed-shift
run_leg() {
  local out="$1" bench="$2" backend="$3" disclosure="$4"
  local label="$bench $backend $disclosure"
  local warmup_args=()
  [ -n "$WARMUP" ] && warmup_args=(--warmup "$WARMUP")

  local llm_create=(
    --llm-provider "$PROVIDER" --llm-model "$MODEL"
  )
  [ -n "${BASE_URL:-}" ] && llm_create+=(--llm-base-url "$BASE_URL")
  [ -n "${EXTRA_BODY:-}" ] && llm_create+=(--llm-extra-body "$EXTRA_BODY")
  [ -n "${KERNEL_LLM_API_KEY_ENV:-}" ] && llm_create+=(--llm-api-key-env "$KERNEL_LLM_API_KEY_ENV")

  local kernel_args=()
  if _kernel_override; then
    kernel_args=(
      --kernel-llm-provider "$KERNEL_LLM_PROVIDER" --kernel-llm-model "$KERNEL_LLM_MODEL"
    )
    [ -n "${KERNEL_LLM_BASE_URL:-}" ] && kernel_args+=(--kernel-llm-base-url "$KERNEL_LLM_BASE_URL")
    [ -n "${KERNEL_LLM_API_KEY_ENV:-}" ] && kernel_args+=(--kernel-llm-api-key-env "$KERNEL_LLM_API_KEY_ENV")
    [ -n "${KERNEL_LLM_EXTRA_BODY:-}" ] && kernel_args+=(--kernel-llm-extra-body "$KERNEL_LLM_EXTRA_BODY")
  fi

  local sara=(
    --benchmark "$bench" --provider "$PROVIDER" --model "$MODEL"
    --base-url "$BASE_URL" --api-key "$API_KEY"
    --budget "$BUDGET" --seed "$SEED" --root "$out"
    --acqf "$ACQF" --extra-body "$EXTRA_BODY"
  )
  [ -n "$WARMUP" ] && sara+=(--warmup "$WARMUP")

  local baseline=(
    --benchmark "$bench" --budget "$BUDGET" --seed "$SEED"
    --root "$out" "${warmup_args[@]+"${warmup_args[@]}"}"
  )
  case "$disclosure" in
    blind) ;;
    revealed) baseline+=(--reveal) ;;
    revealed-shift) baseline+=(--reveal --shift) ;;
    *)
      echo "error: disclosure must be blind, revealed, or revealed-shift (got '$disclosure')" >&2
      return 1
      ;;
  esac

  local shift_args=()
  [ "$disclosure" = "revealed-shift" ] && shift_args=(--shift)

  case "$backend" in
    vanilla)
      skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_baseline \
        "${baseline[@]}" --policy vanilla
      ;;
    turbo)
      skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_baseline \
        "${baseline[@]}" --policy turbo
      ;;
    cake)
      skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_baseline \
        "${baseline[@]}" --policy cake "${llm_create[@]}" "${kernel_args[@]+"${kernel_args[@]}"}"
      ;;
    sara-lenz)
      if [ "$disclosure" = "blind" ]; then
        skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_test \
          "${sara[@]}" --surrogate fixed
      else
        skip_or_run "$out" "$label" python3 -m benchmarks.run_noblind_test \
          "${sara[@]}" --surrogate fixed "${shift_args[@]+"${shift_args[@]}"}" \
          --context-variant "$CONTEXT_VARIANT"
      fi
      ;;
    sara-lenz-cake)
      if [ "$disclosure" = "blind" ]; then
        skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_test \
          "${sara[@]}" --surrogate cake "${kernel_args[@]+"${kernel_args[@]}"}"
      else
        skip_or_run "$out" "$label" python3 -m benchmarks.run_noblind_test \
          "${sara[@]}" --surrogate cake "${shift_args[@]+"${shift_args[@]}"}" \
          --context-variant "$CONTEXT_VARIANT" "${kernel_args[@]+"${kernel_args[@]}"}"
      fi
      ;;
    sara-only)
      if [ "$disclosure" = "blind" ]; then
        skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_test \
          "${sara[@]}" --no-lenz
      else
        skip_or_run "$out" "$label" python3 -m benchmarks.run_noblind_test \
          "${sara[@]}" --no-lenz "${shift_args[@]+"${shift_args[@]}"}" \
          --context-variant "$CONTEXT_VARIANT"
      fi
      ;;
    sara-lenz-turbo)
      # Region slot pinned to TuRBO under the agent, mirroring how
      # sara-lenz-cake pins the surrogate slot -- tests whether the surrogate
      # slot's high variance under an agent (see the blog's Hartmann6 vs
      # Ackley10 discussion) is CAKE-specific or a general pinned-plugin effect.
      if [ "$disclosure" = "blind" ]; then
        skip_or_run "$out" "$label" python3 -m benchmarks.run_blind_test \
          "${sara[@]}" --surrogate fixed --region turbo
      else
        skip_or_run "$out" "$label" python3 -m benchmarks.run_noblind_test \
          "${sara[@]}" --surrogate fixed --region turbo "${shift_args[@]+"${shift_args[@]}"}" \
          --context-variant "$CONTEXT_VARIANT"
      fi
      ;;
    sara-lenz-pibo)
      # Prior slot pinned to a scripted pi-BO belief under the agent. Belief
      # fixtures are deterministic and currently only defined for bolt_lora
      # (benchmarks/priors.py), matching the scripted `pibo` baseline.
      if [ "$disclosure" != "blind" ] && [ "$bench" = "bolt_lora" ]; then
        skip_or_run "$out" "$label" python3 -m benchmarks.run_noblind_test \
          "${sara[@]}" --surrogate fixed --prior-fixture "bolt-$CONTEXT_VARIANT" \
          "${shift_args[@]+"${shift_args[@]}"}" --context-variant "$CONTEXT_VARIANT"
      else
        echo "error: sara-lenz-pibo currently requires bolt_lora (revealed) -- belief fixtures are bolt-only" >&2
        return 1
      fi
      ;;
    *)
      echo "error: unknown backend '$backend' (vanilla, cake, turbo, sara-lenz, sara-lenz-cake, sara-only, sara-lenz-turbo, sara-lenz-pibo)" >&2
      return 1
      ;;
  esac
}

expand_backends() {
  local raw="$1" default="$2"
  if [ "$raw" = "all" ]; then
    raw="$default"
  fi
  echo "$raw" | tr ',' ' '
}

plot_root() {
  local root="$1" title="$2"
  if [ "$LIST" = "1" ]; then
    echo "PLOT  python3 -m benchmarks.plot_compare --root $root"
    return 0
  fi
  python3 -m benchmarks.plot_compare --root "$root" --title "$title"
}
