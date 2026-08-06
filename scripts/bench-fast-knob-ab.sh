#!/usr/bin/env bash
# bench-fast-knob-ab.sh — fast runtime-knob A/B on the shipped champion binary.
#
# New direction: instead of source edits (leads 1-4 tried), sweep the ~50
# GGML_SYCL_* runtime knobs that are ALREADY compiled into the champion binary
# (probe-verified, see scripts/bench-sweep-runtime-knobs.sh) using the EXACT
# official bench flags, one combined pp512+tg128 run per variant.
#
# Official scoring env only propagates DISABLE_GRAPH + DISABLE_DNN; the
# ENABLE_* family (graph, flash-attn, fusion, opt, VMM, DMMV_X, …) is
# un-explored on this binary. A knob that lifts tg without touching the
# broken dual_down/multitoken source paths ships as a pure env change.
#
# Usage: bash scripts/bench-fast-knob-ab.sh [--bin DIR] [--reps N]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

# Prefer the actual llama-bench executable (LX_LLAMA_BENCH); LX_BIN may be a dir
BIN="${LX_LLAMA_BENCH:-$LX_BIN}"
[ -f "$BIN" ] || BIN="$BIN/llama-bench"
BIN_DIR="$(dirname "$BIN")"
REPS="${LX_REPS:-2}"
MODEL="${LX_MODEL:?}"

# Rogue-server contention guard (fast path; full lock still applied below)
if pgrep -f "llama-server" >/dev/null 2>&1; then
  echo "WARN: llama-server running — kill it first for clean measurements" >&2
fi

lx_gpu_lock_enter "knob-ab" || exit $?
trap 'lx_gpu_lock_leave' EXIT

# Exact official window/flags (mirrors bench-serial.sh COMMON for this build:
# no -c, no -tb; window is -p/-n only)
FA_ARGS=()
COMMON=(
  -m "$MODEL"
  -ngl 99
  --n-cpu-moe 0
  --split-mode layer
  --main-gpu 0
  --tensor-split 0
  --device auto
  -t 16
  --cpu-mask 0x0
  --cpu-strict 0
  -b 2048 -ub 2048
  -ctk f16 -ctv f16
  --no-kv-offload 0 --no-op-offload 0 --no-host 0
  "${FA_ARGS[@]}"
  -r "$REPS" -d 0
  --prio 0
  --load-mode mmap
  --poll 0
  --delay 0
  -o json
)

# Variants: name::env-assignments (comma-separated)
# default = official scoring env (DISABLE_GRAPH=1 DISABLE_DNN=1 from env.sh)
VARIANTS=(
  "default::"
  "graph_on::GGML_SYCL_ENABLE_GRAPH=1"
  "flash_attn::GGML_SYCL_ENABLE_FLASH_ATTN=1"
  "fa_onednn::GGML_SYCL_FA_ONEDNN=1"
  "fusion::GGML_SYCL_ENABLE_FUSION=1"
  "opt::GGML_SYCL_ENABLE_OPT=1"
  "vmm::GGML_SYCL_ENABLE_VMM=1"
  "f16::GGML_SYCL_F16=1"
  "dmmv_x2::GGML_SYCL_DMMV_X=2"
  "async_mem::GGML_SYCL_USE_ASYNC_MEM_OP=1"
  "lv0_api::GGML_SYCL_USE_LEVEL_ZERO_API=1"
  "prioritize_dmmv::GGML_SYCL_PRIORITIZE_DMMV=1"
)

echo "== knob A/B on $BIN_DIR (sha $(sha256sum "$BIN" | awk '{print $1}' | cut -c1-12 2>/dev/null))"
echo "== model: $MODEL"
echo "== flags: -pg 512,128 reps=$REPS"

declare -A RES
for v in "${VARIANTS[@]}"; do
  name="${v%%::*}"; envs="${v#*::}"
  echo "--- VARIANT $name (${envs:-official env})"
  cmd=(env ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}" ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}")
  if [ -n "$envs" ]; then
    IFS=',' read -r -a pairs <<< "$envs"
    for pair in "${pairs[@]}"; do cmd+=("$pair"); done
  fi
  out="$( "${cmd[@]}" "$BIN" "${COMMON[@]}" -pg 512,128 2>/dev/null )"
  # llama-bench json: [{"test_id":"pp512","avg_ts":...,"n_prompt":512},...]
  # match on n_prompt/n_gen (like bench-serial.sh parse_ts); field names vary
  pp="$(printf '%s' "$out" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin)
 rows = d["results"] if isinstance(d,dict) else d
 for r in rows:
  np=r.get("n_prompt"); ng=r.get("n_gen")
  if ng in (0,"0",None) and np: print(r.get("avg_ts",0)); break
except Exception: print("ERR")' 2>/dev/null || echo ERR)"
  tg="$(printf '%s' "$out" | python3 -c 'import json,sys
try:
 d=json.load(sys.stdin)
 rows = d["results"] if isinstance(d,dict) else d
 for r in rows:
  np=r.get("n_prompt"); ng=r.get("n_gen")
  if np in (0,"0",None) and ng: print(r.get("avg_ts",0)); break
except Exception: print("ERR")' 2>/dev/null || echo ERR)"
  echo "  pp512=${pp}  tg128=${tg}"
  RES["$name"]="$pp|$tg"
done

echo
echo "== TABLE (pp512 / tg128 t/s) — default first =="
printf "%-16s %10s %10s\n" variant pp512 tg128
for v in "${VARIANTS[@]}"; do
  name="${v%%::*}"
  IFS='|' read -r pp tg <<< "${RES[$name]}"
  printf "%-16s %10s %10s\n" "$name" "$pp" "$tg"
done
echo "== DONE — winning tg variant can be promoted via env in bench-serial.sh =="
