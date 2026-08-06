#!/bin/bash
# vdr2-cycle.sh — same-window A/B for GGML_SYCL_Q6K_VDR2 (VDR=2 ncols==1 q6_K mmvq).
# Builds BOTH ggml-sycl.cpp and mmvq.cpp from src-lmhead (the probe mechanism only
# rebuilds ggml-sycl.cpp; mmvq.cpp is a separate TU), swaps both objects into the
# src-lmhead-build tree, relinks, then benches ctrl / vdr2 / ctrl in one lock window.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=$ROOT/src-lmhead
BUILD=$ROOT/src-lmhead-build

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib
[ -d /opt/intel/oneapi/compiler/2026.0/bin ] && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2026.0/bin

# 1) build ggml-sycl.cpp probe (as before)
bash "$BUILD/probe-build.sh" >/tmp/vdr2-build.log 2>&1 || { echo "PROBE BUILD FAIL"; tail -20 /tmp/vdr2-build.log; exit 2; }

# 2) build mmvq.cpp with the same defines (compile command from probe-build.sh minus ggml-sycl.cpp)
ICPC=/opt/intel/oneapi/compiler/2026.0/bin/icpx
DEFS=(-DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_SYCL_DNNL=1 -DGGML_SYCL_F16 -DGGML_SYCL_GRAPH -DGGML_SYCL_HOST_MEM_FALLBACK -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API -DGGML_SYCL_WARP_SIZE=16 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_sycl_EXPORTS)
INCS=(-I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/ggml-sycl/.. -I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/../include -isystem /opt/intel/oneapi/compiler/2026.0/include -isystem /opt/intel/oneapi/dnnl/2026.0/include -isystem /opt/intel/oneapi/mkl/2026.0/include)
"$ICPC" "${DEFS[@]}" "${INCS[@]}" -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-unused-function -Wno-narrowing -fsycl -DMKL_ILP64 \
  -o "$BUILD/probe-mmvq.o" -c "$SRC/ggml/src/ggml-sycl/mmvq.cpp" >>/tmp/vdr2-build.log 2>&1 \
  || { echo "MMVQ BUILD FAIL"; tail -20 /tmp/vdr2-build.log; exit 3; }
echo "[vdr2] mmvq.cpp probe built"

# 3) locate both objects in the build tree and swap
OBJDIR=$(dirname "$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)")
[ -n "$OBJDIR" ] || { echo "no object dir"; exit 4; }
MMVQ_OBJ=$(find "$BUILD" -name 'mmvq.cpp.o' | head -1)
[ -n "$MMVQ_OBJ" ] || { echo "no mmvq.cpp.o"; exit 5; }
cp "$BUILD/probe-ggml-sycl.o" "$OBJDIR/ggml-sycl.cpp.o"
cp "$BUILD/probe-mmvq.o" "$MMVQ_OBJ"
echo "[vdr2] swapped ggml-sycl.cpp.o + $MMVQ_OBJ"

# 4) relink via link.txt (cwd = the tree that owns link.txt)
LINK="$OBJDIR/link.txt"
if [ -f "$LINK" ]; then
  ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >>/tmp/vdr2-build.log 2>&1 \
    || { echo "RELINK FAIL"; tail -20 /tmp/vdr2-build.log; exit 6; }
else
  cmake --build "$BUILD" --target ggml-sycl -j32 >>/tmp/vdr2-build.log 2>&1 || { echo "CMAKE FAIL"; exit 6; }
fi
[ -x "$BUILD/bin/llama-bench" ] || { echo "no llama-bench after relink"; exit 7; }
echo "[vdr2] relinked — A/B arms:"
strings "$BUILD/bin/llama-bench" | grep -c GGML_SYCL_Q6K_VDR2 || true

# 5) same-window sandwich: ctrl / vdr2 / ctrl
ARGS=("$@")
if [ $# -eq 0 ]; then ARGS=(ctrl-a:- vdr2-1:1 ctrl-b:-); fi
bash "$ROOT/scripts/interleave-source-ab.sh" "${ARGS[@]}" | tee /tmp/vdr2-ab.log
