#!/usr/bin/env bash
# Fresh decode wall decomposition on the CURRENT champion build (post-K-fuse),
# plus probes the 2026-08-09 script did not cover:
#   bit 16 = lm_head-only skip (isolates the 100352-row GEMV)
#   GGML_SYCL_DIAG_SKIP_TINY_N=1 = skip GEMM exec for ncols<=1 (all decode MUL_MAT)
#   control re-run at the end for drift bounds
set -u
LX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$LX_ROOT/benchmark/kernel/build/bin"
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
OUT="$LX_ROOT/results/lx-decode-wall-v2-$(date +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

# shellcheck disable=SC1091
source "$LX_ROOT/scripts/lib-gpu-lock.sh"
lx_gpu_lock_enter "decode-wall-v2"
trap 'lx_gpu_lock_leave' EXIT

ARGS=(-m "$MODEL" -ngl 99 -t 16 -ub 2048 -b 2048 -p 0 -n 128 -r 5 -ctk f16 -ctv f16 --poll 50)

echo "bin sha: $(sha256sum "$BIN/llama-bench" | awk '{print $1}')" | tee "$OUT/results.txt"

run_one() {
  local label="$1"; shift
  env "$@" "$BIN/llama-bench" "${ARGS[@]}" > "$OUT/$label.log" 2>&1
  local ts ms
  ts=$(grep -A1 "| tg128" "$OUT/$label.log" | tail -1 | awk -F'|' '{gsub(/ /,"",$5); print $5}')
  ms=$(grep -A1 "| tg128" "$OUT/$label.log" | tail -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  echo "$label tg=$ts ms=$ms" | tee -a "$OUT/results.txt"
}

run_one skip0
run_one skip1  GGML_SYCL_DIAG_SKIP_DECODE=1
run_one skip2  GGML_SYCL_DIAG_SKIP_DECODE=2
run_one skip4  GGML_SYCL_DIAG_SKIP_DECODE=4
run_one skip8  GGML_SYCL_DIAG_SKIP_DECODE=8
run_one skip16 GGML_SYCL_DIAG_SKIP_DECODE=16
run_one skip15 GGML_SYCL_DIAG_SKIP_DECODE=15
run_one tiny1  GGML_SYCL_DIAG_SKIP_TINY_N=1
run_one skip0b

echo "OUT=$OUT"
