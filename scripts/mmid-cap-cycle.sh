#!/bin/bash
# mmid-cap-cycle.sh — same-window ctrl / MMID_FUSED_MAX_TOKENS=512 / ctrl A/B.
# Extends the fused per-token mul_mat_id path (ggml-sycl.cpp ne12_cap, default 64)
# so pp512 prefill routed-expert down-proj runs the fused per-token MMVQ path
# instead of the counting-sort grouped-GEMM path. Bit-exact per-token kernel.
# Usage: bash scripts/mmid-cap-cycle.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=$ROOT/src-lmhead
BUILD=$ROOT/src-lmhead-build

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:/opt/intel/oneapi/compiler/2026.0/bin:${LD_LIBRARY_PATH:-}

LOG=/tmp/mmid-cap-build.log
# 1) compile the ggml-sycl.cpp probe (edit is in this TU only)
bash "$BUILD/probe-build.sh" >"$LOG" 2>&1 || { echo "PROBE BUILD FAIL"; tail -20 "$LOG"; exit 3; }

# 2) swap the probe object into the build tree and relink
OBJDIR=$(dirname "$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)")
[ -n "$OBJDIR" ] || { echo "no object dir"; exit 5; }
cp "$BUILD/probe-ggml-sycl.o" "$OBJDIR/ggml-sycl.cpp.o"
echo "[mmid-cap] swapped ggml-sycl.cpp.o"
LINK="$OBJDIR/link.txt"
if [ -f "$LINK" ]; then
  ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >>"$LOG" 2>&1 \
    || { echo "RELINK FAIL"; tail -20 "$LOG"; exit 6; }
else
  cmake --build "$BUILD" --target ggml-sycl -j32 >>"$LOG" 2>&1 || { echo "CMAKE FAIL"; exit 6; }
fi
[ -x "$BUILD/bin/llama-bench" ] || { echo "no llama-bench after relink"; exit 7; }

# 3) golden smoke on the probe lib (default cap 64 = champion behavior, bit-exact)
bash "$ROOT/scripts/golden-smoke.sh" >"$ROOT/results/mmid-cap-golden.log" 2>&1 \
  || { echo "GOLDEN FAIL"; tail -20 "$ROOT/results/mmid-cap-golden.log"; exit 8; }
echo "[mmid-cap] GOLDEN OK"

# 4) official-geometry bench, three arms in one lock window
BENCH="$BUILD/bin/llama-bench"
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
[ -f "$MODEL" ] || { echo "missing model: $MODEL"; exit 9; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/results/mmid-cap-$STAMP"
mkdir -p "$OUT"
ARMS=(-m "$MODEL" -ngl 99 -t 16 --n-cpu-moe 0 --split-mode layer --main-gpu 0 --tensor-split 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 --no-kv-offload 0 --no-op-offload 0 --no-host 0 --prio 0 --load-mode mmap --poll 50 -p 512 -n 128 -r 5 -o json)
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1

"$ROOT/scripts/with-gpu-lock" --wait 900 --reason "mmid-cap" -- bash -c "
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1
  echo '=== CTRL-A ==='
  env -u GGML_SYCL_MMID_FUSED_MAX_TOKENS $BENCH ${ARMS[*]} | tee $OUT/ctrl-a.json
  echo '=== MAX_TOKENS=512 ==='
  GGML_SYCL_MMID_FUSED_MAX_TOKENS=512 GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1 $BENCH ${ARMS[*]} | tee $OUT/cand.json
  echo '=== CTRL-B ==='
  env -u GGML_SYCL_MMID_FUSED_MAX_TOKENS $BENCH ${ARMS[*]} | tee $OUT/ctrl-b.json
" || { echo "BENCH WINDOW FAIL"; exit 8; }

echo "[mmid-cap] receipts in $OUT"
for f in ctrl-a cand ctrl-b; do
  echo "-- $f: tg=$(jq -r '.[] | select(.test_id=="tg128") | .avg_ts' $OUT/$f.json 2>/dev/null || echo NOPARSE) pp=$(jq -r '.[] | select(.test_id=="pp512") | .avg_ts' $OUT/$f.json 2>/dev/null || echo NOPARSE)"
done
