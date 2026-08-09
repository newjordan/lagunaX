#!/usr/bin/env bash
# Residual-decomposition probe on the current champion .so (build/bin).
# skip15 removes the four big decode classes (QKV, MoE-ID, dense vec-q, fattn),
# leaving the ~2.9 ms/token residual. DIAG_SKIP_F32 additionally elides the
# F32-src0 MUL_MAT GEMM exec (op_mul_mat_sycl) — at decode that is the MoE
# router GEMV (gate_inp, f32 2048x256) per layer. Two runs + a control for
# drift bounds; same session, same binary.
set -u
LX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$LX_ROOT/benchmark/kernel/build/bin"
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
OUT="$LX_ROOT/results/lx-decode-residual-v2-$(date +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

# shellcheck disable=SC1091
source "$LX_ROOT/scripts/lib-gpu-lock.sh"
lx_gpu_lock_enter "decode-residual-v2"
trap 'lx_gpu_lock_leave' EXIT

ARGS=(-m "$MODEL" -ngl 99 -t 16 -ub 2048 -b 2048 -p 0 -n 128 -r 5 -ctk f16 -ctv f16 --poll 50)

echo "bin sha: $(sha256sum "$BIN/llama-bench" | awk '{print $1}')" | tee "$OUT/results.txt"

run_one() {
  local label="$1"; shift
  env "$@" "$BIN/llama-bench" "${ARGS[@]}" > "$OUT/$label.log" 2>&1
  local ts
  ts=$(grep -m1 'tg128' "$OUT/$label.log" | awk -F'|' '{print $9}' | tr -d ' ' | sed 's/±.*//')
  echo "$label tg=$ts" | tee -a "$OUT/results.txt"
}

run_one skip0
run_one skip15   GGML_SYCL_DIAG_SKIP_DECODE=15
run_one skip15f  GGML_SYCL_DIAG_SKIP_DECODE=15 GGML_SYCL_DIAG_SKIP_F32=1
run_one skip0b

echo "OUT=$OUT"
