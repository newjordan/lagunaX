#!/bin/bash
# vdrN-cycle.sh — same-window ctrl / VDR-N / ctrl A/B for the ncols==1 q6_K
# mmvq path (GGML_SYCL_Q6K_VDR=N, bit-exact work-distribution depth).
# Usage: bash scripts/vdrN-cycle.sh <N>   (N in 2|4|8)
# Rebuilds ggml-sycl.cpp + mmvq.cpp probes from src-lmhead, swaps into the
# src-lmhead-build tree, relinks, then runs the OFFICIAL pp512/tg128 bench
# (champion flags, reps=5) three times in ONE lock window: ctrl / VDR-N / ctrl.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=$ROOT/src-lmhead
BUILD=$ROOT/src-lmhead-build
VDR="${1:?usage: vdrN-cycle.sh <2|4|8>}"
case "$VDR" in 2|4|8) ;; *) echo "VDR must be 2|4|8"; exit 2 ;; esac

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
# PREPEND (never clobber): setvars adds libumf etc. that Level Zero needs to
# discover the device; a clobbering export silently kills discovery (rc=134).
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:/opt/intel/oneapi/compiler/2026.0/bin:${LD_LIBRARY_PATH:-}

# 1) build ggml-sycl.cpp probe (as committed in vdr2-cycle.sh)
bash "$BUILD/probe-build.sh" >/tmp/vdrN-build.log 2>&1 || { echo "PROBE BUILD FAIL"; tail -20 /tmp/vdrN-build.log; exit 3; }

# 2) build mmvq.cpp with the same defines/flags
ICPC=/opt/intel/oneapi/compiler/2026.0/bin/icpx
DEFS=(-DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_SYCL_DNNL=1 -DGGML_SYCL_F16 -DGGML_SYCL_GRAPH -DGGML_SYCL_HOST_MEM_FALLBACK -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API -DGGML_SYCL_WARP_SIZE=16 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_sycl_EXPORTS)
INCS=(-I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/ggml-sycl/.. -I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/../include -isystem /opt/intel/oneapi/compiler/2026.0/include -isystem /opt/intel/oneapi/dnnl/2026.0/include -isystem /opt/intel/oneapi/mkl/2026.0/include)
"$ICPC" "${DEFS[@]}" "${INCS[@]}" -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-unused-function -Wno-narrowing -fsycl -DMKL_ILP64 \
  -o "$BUILD/probe-mmvq.o" -c "$SRC/ggml/src/ggml-sycl/mmvq.cpp" >>/tmp/vdrN-build.log 2>&1 \
  || { echo "MMVQ BUILD FAIL"; tail -20 /tmp/vdrN-build.log; exit 4; }

# 3) swap both objects into the build tree and relink
OBJDIR=$(dirname "$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)")
[ -n "$OBJDIR" ] || { echo "no object dir"; exit 5; }
MMVQ_OBJ=$(find "$BUILD" -name 'mmvq.cpp.o' | head -1)
[ -n "$MMVQ_OBJ" ] || { echo "no mmvq.cpp.o"; exit 5; }
cp "$BUILD/probe-ggml-sycl.o" "$OBJDIR/ggml-sycl.cpp.o"
cp "$BUILD/probe-mmvq.o" "$MMVQ_OBJ"
echo "[vdrN] swapped ggml-sycl.cpp.o + mmvq.cpp.o"

LINK="$OBJDIR/link.txt"
if [ -f "$LINK" ]; then
  ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >>/tmp/vdrN-build.log 2>&1 \
    || { echo "RELINK FAIL"; tail -20 /tmp/vdrN-build.log; exit 6; }
else
  cmake --build "$BUILD" --target ggml-sycl -j32 >>/tmp/vdrN-build.log 2>&1 || { echo "CMAKE FAIL"; exit 6; }
fi
[ -x "$BUILD/bin/llama-bench" ] || { echo "no llama-bench after relink"; exit 7; }

# 4) official-geometry bench, three arms in one lock window
BENCH="$BUILD/bin/llama-bench"
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
[ -f "$MODEL" ] || { echo "missing model: $MODEL"; exit 9; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/results/vdrN-$VDR-$STAMP"
mkdir -p "$OUT"

ARMS=(-m "$MODEL" -ngl 99 -t 16 --n-cpu-moe 0 --split-mode layer --main-gpu 0 --tensor-split 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 --no-kv-offload 0 --no-op-offload 0 --no-host 0 --prio 0 --load-mode mmap --poll 50 -p 512 -n 128 -r 5 -o json)
# NOTE: this llama-bench build has no -tb / --sycl-disable-* flags; graph+dnn
# disable are env vars (GGML_SYCL_DISABLE_GRAPH/DNN), matching the champion run.

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1

"$ROOT/scripts/with-gpu-lock" --wait 900 --reason "vdrN-$VDR" -- bash -c "
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1
  echo '=== CTRL-A ==='
  env -u GGML_SYCL_Q6K_VDR -u GGML_SYCL_Q6K_VDR2 $BENCH ${ARMS[*]} | tee $OUT/ctrl-a.json
  echo '=== VDR=$VDR ==='
  GGML_SYCL_Q6K_VDR=$VDR $BENCH ${ARMS[*]} | tee $OUT/vdr.json
  echo '=== CTRL-B ==='
  env -u GGML_SYCL_Q6K_VDR -u GGML_SYCL_Q6K_VDR2 $BENCH ${ARMS[*]} | tee $OUT/ctrl-b.json
" || { echo "BENCH WINDOW FAIL"; exit 8; }

echo "[vdrN] receipts in $OUT"
for f in ctrl-a vdr ctrl-b; do
  echo "-- $f: $(jq -r '.[] | select(.test_id=="tg128") | .avg_ts' $OUT/$f.json 2>/dev/null || echo NOPARSE)"
done
