#!/usr/bin/env bash
# Build xmx-dequant-gemm. AOT for BMG first (catches joint_matrix codegen issues
# without touching the GPU); falls back to spir64 JIT if the AOT target is
# unavailable. No GPU lock needed to build.
set -uo pipefail
cd "$(dirname "$0")"
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

FLAGS=(-fsycl -O3 -std=c++17 -qmkl ${XMX_EXTRA_FLAGS:-})
SRC=main.cpp
OUT=xmx-dequant-gemm

echo "== AOT build (intel_gpu_bmg_g21) =="
if icpx "${FLAGS[@]}" -fsycl-targets=intel_gpu_bmg_g21 "$SRC" -o "$OUT" 2>build.log; then
  echo "built $OUT (AOT bmg_g21)"
  exit 0
fi
echo "AOT failed (see build.log tail below); falling back to spir64 JIT" >&2
tail -5 build.log >&2
if icpx "${FLAGS[@]}" "$SRC" -o "$OUT" 2>>build.log; then
  echo "built $OUT (spir64 JIT)"
  exit 0
fi
echo "BUILD FAILED — build.log:" >&2
tail -40 build.log >&2
exit 1
