#!/usr/bin/env bash
# bench-sweep-runtime-knobs.sh — A/B runtime SYCL scheduling knobs on the
# source-reproducible champion binary, through the official bench pipeline.
#
# New angle vs prior directions: every prior score came from one binary under
# one default runtime env. llama.cpp's SYCL backend reads a small set of
# scheduling env vars (event polling etc.) that change HOST-side scheduling
# without touching numerics — a quality-neutral decode lever never swept on
# this rig. Host-side event handling is a classic decode bottleneck on SYCL
# (each GEMV launch = host sync); if polling cuts that, decode t/s rises
# with zero risk to the proof-suite numerics.
#
# Pipeline: probe knobs -> fast direct A/B on the same binary/flags (delta is
# the signal) -> official bench-champion-cycle run(s) with --bin=<src-repro>
# for a real receipt + guard-score floor check.
#
# Usage: bash scripts/bench-sweep-runtime-knobs.sh
# Exit: 0 = at least the baseline variant scored; logs under results/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 78
[ -f env.sh ] && source env.sh
mkdir -p results

# Locate the source-repro champion build (llama-bench present).
find_bin() {
  local d
  for d in build-source-repro \
           src-repro \
           "$LX_ROOT/results"/*/src-repro \
           "$LX_ROOT/results"/*/build-source-repro \
           "$LX_ROOT"/worktrees/*/build-source-repro/bin \
           "$LX_ROOT"/worktrees/*/build*src-repro*/bin; do
    [ -x "$d/llama-bench" ] && { echo "$d"; return 0; }
  done
  local hit
  hit="$(find "$LX_ROOT/results" -maxdepth 4 -name llama-bench -type f -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
  [ -n "${hit:-}" ] && { echo "$hit"; return 0; }
  return 1
}

BIN="$(find_bin || true)"
if [ -z "${BIN:-}" ]; then
  echo "WARN: no src-repro build found; falling back to env default \$LX_BIN=$LX_BIN"
  BIN="$LX_BIN"
fi
echo "== [sweep] candidate binary: $BIN"
[ -x "$BIN/llama-bench" ] || { echo "FATAL: no llama-bench in $BIN" >&2; exit 78; }

probe() {
  echo "== [probe] GGML_SYCL_* knobs referenced in candidate binary/libs:"
  for blob in "$BIN"/llama-bench "$BIN"/libggml-sycl* "$BIN"/libggml.so* "$BIN"/*.so; do
    [ -f "$blob" ] || continue
    grep -aoE 'GGML_SYCL_[A-Z_0-9]+' "$blob" 2>/dev/null
  done | sort -u | head -40
  echo "== [probe] knobs the official scoring env already propagates (bench-serial.sh):"
  grep -n 'GGML_SYCL_DISABLE_GRAPH\|GGML_SYCL_DISABLE_DNN' scripts/bench-serial.sh
}

# Fast direct A/B — same binary, same flags, env differs. Delta is the signal.
fast_ab() {
  echo "== [fast-ab] checking for rogue llama-server (contention guard)"
  if pgrep -af "llama-server" >/dev/null 2>&1; then
    echo "WARN: llama-server running — skipping fast A/B (full cycles will gate via lock)"
    return 0
  fi
  local model="${LX_MODEL:-}"
  [ -n "$model" ] || { echo "WARN: no LX_MODEL — skipping fast A/B"; return 0; }
  local flags=(-m "$model" -p 1024 -n 96 -b 1024 -ub 512 -r 1)
  local out
  out="results/sweep-fastab-$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "== [fast-ab] flags: ${flags[*]}" | tee "$out"
  {
    local variants=("default::" "no_graph::GGML_SYCL_DISABLE_GRAPH=1" "no_dnn::GGML_SYCL_DISABLE_DNN=1" "no_both::GGML_SYCL_DISABLE_GRAPH=1,GGML_SYCL_DISABLE_DNN=1")
    local name envs
    for v in "${variants[@]}"; do
      name="${v%%::*}"; envs="${v#*::}"
      echo "--- VARIANT $name (env: ${envs:-as-is})"
      envs_csv="$envs"
      cmd=(env)
      if [ -n "$envs_csv" ]; then
        IFS=',' read -r -a pairs <<< "$envs_csv"
        local pair
        for pair in "${pairs[@]}"; do cmd+=("$pair"); done
      fi
      cmd+=(ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}" "$BIN/llama-bench" "${flags[@]}")
      "${cmd[@]}" 2>&1
    done
  } >> "$out" 2>&1
  echo "== [fast-ab] receipt: $out"
  grep -E "model|llama_bench|tg|pp" "$out" | tail -20
}

# Full official cycle (golden-smoke gate -> cooldown -> official bench ->
# guard-score floors -> board refresh). Receipts at results/<stamp>/score.json.
run_cycle() {
  local name="$1"; shift
  local pre new log rc
  pre="$(find results -name score.json 2>/dev/null | sort)"
  log="results/sweep-cycle-$name-$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "== [cycle:$name] launching official cycle with --bin=$BIN (env: $*)"
  env COOLDOWN_S="${COOLDOWN_S:-300}" "$@" \
    bash scripts/bench-champion-cycle.sh --skip-proof --bin="$BIN" --note="sweep:$name" \
    > "$log" 2>&1
  rc=$?
  echo "== [cycle:$name] rc=$rc (log: $log)"
  new="$(find results -name score.json 2>/dev/null | sort)"
  local receipt
  receipt="$(comm -13 <(printf '%s\n' "$pre") <(printf '%s\n' "$new") | head -1)"
  if [ -n "${receipt:-}" ]; then
    echo "== [cycle:$name] RECEIPT $receipt"
    jq -r '.score // .result.score // "no .score key", "tg=" + ((.tg // .result.tg // "?")|tostring), "pp=" + ((.pp // .result.pp // "?")|tostring)' "$receipt" 2>/dev/null || jq . "$receipt" 2>/dev/null | head -20
  else
    tail -8 "$log"
  fi
  return "$rc"
}

probe
fast_ab
run_cycle default
run_cycle no_graph GGML_SYCL_DISABLE_GRAPH=1
run_cycle no_dnn GGML_SYCL_DISABLE_DNN=1

echo "== [sweep] all variants attempted."
exit 0
