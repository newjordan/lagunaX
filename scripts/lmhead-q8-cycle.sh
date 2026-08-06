#!/bin/bash
# lmhead-q8-cycle.sh — the last unburned lm_head lever on the B70 decode path:
# a one-time q6_K -> q8_0 pre-convert of the fused lm_head weights, so every
# per-token GEMV reads already-converted int8 values instead of re-dequantizing
# q6_K blocks inside the reorder dot (open lead 3: ~130 us/token ceiling, ~4x
# off the 475 GB/s effective-BW bound; finding 18 proves the real kernel is the
# addend-bearing reorder path; finding 20 proves load-ORDER is null, so the
# untouched axis is load-FORMAT).
#
# Payload: results/lmhead-q8/q6k-q8.patch (authored against the live mmvq.cpp,
# two hunks: (1) conversion machinery + cache, (2) q8_0 dispatch gate in the
# q6_K reorder branch, active only when GGML_SYCL_LMHEAD_Q8=1 AND the fused
# addend path is live). Champion-bitexact with the gate OFF.
#
# Cycle: validate payload -> apply -> compile probe object (probe-build.sh) ->
# swap + relink -> golden-smoke WITH gate ON -> [BENCH=1] same-window 3-arm A/B.
# Requires the finding-22 worktree rules: apply via patch --forward, revert via
# git apply -R, NEVER git checkout (write-denied base repo index.lock).

set -u
LX="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$LX/src-lmhead"
MMVQ="$SRC/ggml/src/ggml-sycl/mmvq.cpp"
BUILD="$LX/src-lmhead-build"
PATCH="$LX/results/lmhead-q8/q6k-q8.patch"
GATE="GGML_SYCL_LMHEAD_Q8"
mkdir -p "$LX/results/lmhead-q8"

[ -f "$PATCH" ] || { echo "[lmhead-q8] missing payload $PATCH"; exit 2; }

# --- validate payload against live tree ------------------------------------
( cd "$SRC" && git apply --check "$PATCH" 2>/dev/null || patch --dry-run --forward -p1 < "$PATCH" >/dev/null 2>&1 ) \
  || { echo "[lmhead-q8] PAYLOAD VALIDATION FAIL"; exit 21; }
echo "[lmhead-q8] payload validated against live mmvq.cpp"

# idempotence guard: refuse to double-apply
if grep -q "GGML_SYCL_LMHEAD_Q8" "$MMVQ"; then
  echo "[lmhead-q8] gate already present in mmvq.cpp — refusing re-apply (git apply -R results/lmhead-q8/q6k-q8.patch first)"; exit 20
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[lmhead-q8] DRY_RUN=1: payload validated; stopping before apply/build."
  exit 0
fi

# --- apply payload -----------------------------------------------------------
( cd "$SRC" && git apply "$PATCH" 2>/dev/null || patch --forward -p1 < "$PATCH" ) \
  || { echo "[lmhead-q8] APPLY FAIL"; exit 21; }
trap 'cd "$SRC" && ( git apply -R "$PATCH" || patch -R -p1 < "$PATCH" ) >/dev/null 2>&1 || true' EXIT
echo "[lmhead-q8] applied $PATCH"
grep -c "lmhead_q6_to_q8_kernel" "$MMVQ" || true

# --- compile probe objects (BOTH TUs), swap, relink (mirrors vdrN-cycle) -----
# CRITICAL (vdr2-cycle header): the probe mechanism only rebuilds ggml-sycl.cpp;
# mmvq.cpp is a SEPARATE TU — the lmhead-prefetch cycle skipped it and silently
# measured champion-vs-champion. This cycle builds BOTH and swaps BOTH.
[ -f "$BUILD/probe-build.sh" ] || { echo "missing $BUILD/probe-build.sh"; exit 2; }
[ -x "$BUILD/bin/llama-bench" ] || { echo "missing champion llama-bench"; exit 2; }
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:/opt/intel/oneapi/compiler/2026.0/bin:${LD_LIBRARY_PATH:-}
if [ -z "${SKIP_COMPILE:-}" ] || [ ! -f "$BUILD/probe-ggml-sycl.o" ]; then
  bash "$BUILD/probe-build.sh" >/tmp/lmhead-q8-build.log 2>&1 \
    || { echo "BUILD FAIL:"; tail -20 /tmp/lmhead-q8-build.log; exit 3; }
fi
ICPC=/opt/intel/oneapi/compiler/2026.0/bin/icpx
DEFS=(-DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_SYCL_DNNL=1 -DGGML_SYCL_F16 -DGGML_SYCL_GRAPH -DGGML_SYCL_HOST_MEM_FALLBACK -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API -DGGML_SYCL_WARP_SIZE=16 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_sycl_EXPORTS)
INCS=(-I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/ggml-sycl/.. -I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/../include -isystem /opt/intel/oneapi/compiler/2026.0/include -isystem /opt/intel/oneapi/dnnl/2026.0/include -isystem /opt/intel/oneapi/mkl/2026.0/include)
if [ -z "${SKIP_COMPILE:-}" ] || [ ! -f "$BUILD/probe-mmvq.o" ]; then
  "$ICPC" "${DEFS[@]}" "${INCS[@]}" -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-unused-function -Wno-narrowing -fsycl -DMKL_ILP64 \
    -o "$BUILD/probe-mmvq.o" -c "$SRC/ggml/src/ggml-sycl/mmvq.cpp" >>/tmp/lmhead-q8-build.log 2>&1 \
    || { echo "MMVQ BUILD FAIL:"; tail -30 /tmp/lmhead-q8-build.log; exit 4; }
fi
OBJDIR=$(dirname "$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)")
MMVQ_OBJ=$(find "$BUILD" -name 'mmvq.cpp.o' | head -1)
[ -n "$OBJDIR" ] && [ -n "$MMVQ_OBJ" ] || { echo "[lmhead-q8] missing build objects"; exit 5; }
cp "$BUILD/probe-ggml-sycl.o" "$OBJDIR/ggml-sycl.cpp.o"
cp "$BUILD/probe-mmvq.o" "$MMVQ_OBJ"
echo "[lmhead-q8] swapped ggml-sycl.cpp.o + mmvq.cpp.o"
LINK="$OBJDIR/link.txt"
if [ -f "$LINK" ]; then
  ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >/tmp/lmhead-q8-link.log 2>&1 \
    || { echo "RELINK FAIL:"; tail -20 /tmp/lmhead-q8-link.log; exit 5; }
else
  cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/lmhead-q8-link.log 2>&1 \
    || { echo "CMAKE RELINK FAIL:"; tail -20 /tmp/lmhead-q8-link.log; exit 5; }
fi
echo "[lmhead-q8] probe built, swapped, relinked"

# --- golden-smoke WITH the gate ON (quality-invisibility gate) ---------------
if [ "${GOLDEN:-1}" = "1" ]; then
  GATE="$GATE" bash scripts/golden-smoke.sh >/tmp/lmhead-q8-golden.log 2>&1 \
    || { echo "GOLDEN FAIL (gate not quality-neutral):"; tail -20 /tmp/lmhead-q8-golden.log; exit 6; }
  echo "[lmhead-q8] GOLDEN OK"
fi

if [ "${BENCH:-0}" != "1" ]; then
  echo "[lmhead-q8] probe built + golden OK. BENCH=1 to run the same-window 3-arm A/B."
  exit 0
fi

# --- same-window CTRL vs candidate sandwich (official geometry) --------------
echo "[lmhead-q8] arm1 CTRL (gate off) -> arm2 candidate (gate on) -> arm3 CTRL"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "results/lmhead-q8-$STAMP"
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
./scripts/with-gpu-lock --wait -- env -u "$GATE" "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m "$MODEL" \
  > "results/lmhead-q8-$STAMP/ctrl-a.log" 2>&1 || { echo "CTRL-A FAIL"; exit 7; }
./scripts/with-gpu-lock --wait -- env "$GATE"=1 "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m "$MODEL" \
  > "results/lmhead-q8-$STAMP/cand.log" 2>&1 || { echo "CAND FAIL"; exit 7; }
./scripts/with-gpu-lock --wait -- env -u "$GATE" "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m "$MODEL" \
  > "results/lmhead-q8-$STAMP/ctrl-b.log" 2>&1 || { echo "CTRL-B FAIL"; exit 7; }

echo "[lmhead-q8] A/B window complete: results/lmhead-q8-$STAMP/"
echo "[lmhead-q8] next: python3 scripts/ab-aggregate.sh-style parse of the three logs"
exit 0
