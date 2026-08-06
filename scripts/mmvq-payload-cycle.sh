#!/bin/bash
# mmvq-payload-cycle.sh <payload.patch> <label> [gate_env]
#
# Parameterized clone of the proven, committed lmhead-q8-cycle.sh (leads 12/17:
# the lmhead-prefetch 15:36Z A/B was invalidated because probe-build.sh compiles
# ONLY ggml-sycl.cpp while the payload lived in mmvq.cpp — a separate TU — so
# that cycle silently measured champion-vs-champion). This harness builds BOTH
# TUs (probe-build.sh -> probe-ggml-sycl.o, explicit icpx -> probe-mmvq.o),
# swaps both, relinks via link.txt, goldens with the gate ON, then runs the
# same-window ctrl|gate-on|ctrl official-geometry sandwich.
#
# Usage: bash scripts/mmvq-payload-cycle.sh results/<dir>/<payload>.patch ffnout8 GGML_SYCL_FFNOUT_Q8
#   GOLDEN=0 SKIP_COMPILE=1 BENCH=1 overrides mirror the reference cycle.
# Requires the finding-22 worktree rules: apply via git apply||patch --forward,
# revert via git apply -R||patch -R, NEVER git checkout (write-denied base repo).

set -u
LX="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$LX/src-lmhead"
MMVQ="$SRC/ggml/src/ggml-sycl/mmvq.cpp"
BUILD="$LX/src-lmhead-build"
PATCH="${1:?usage: mmvq-payload-cycle.sh <payload.patch> <label> [gate_env]}"
PATCH="$(readlink -f "$PATCH" 2>/dev/null || echo "$PATCH")"
LABEL="${2:?usage: mmvq-payload-cycle.sh <payload.patch> <label> [gate_env]}"
GATE="${3:-GGML_SYCL_${LABEL}}"
OUT="$LX/results/$LABEL"
mkdir -p "$OUT"

[ -f "$PATCH" ] || { echo "[$LABEL] missing payload $PATCH"; exit 2; }

# --- validate payload against live tree ------------------------------------
( cd "$SRC" && git apply --check "$PATCH" 2>/dev/null || patch --dry-run --forward -p1 < "$PATCH" >/dev/null 2>&1 ) \
  || { echo "[$LABEL] PAYLOAD VALIDATION FAIL"; exit 21; }
echo "[$LABEL] payload validated against live mmvq.cpp"

# idempotence guard: refuse to double-apply
if grep -q "$GATE" "$MMVQ"; then
  echo "[$LABEL] gate already present in mmvq.cpp — refusing re-apply (git apply -R $PATCH first)"; exit 20
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[$LABEL] DRY_RUN=1: payload validated; stopping before apply/build."
  exit 0
fi

# --- apply payload -----------------------------------------------------------
( cd "$SRC" && git apply "$PATCH" 2>/dev/null || patch --forward -p1 < "$PATCH" ) \
  || { echo "[$LABEL] APPLY FAIL"; exit 21; }
CHAMP_SO="$LX/results/src-repro-20260806T035656Z/bin/libggml-sycl.so.0.17.0"
restore_champion() {
  cd "$SRC" && ( git apply -R "$PATCH" || patch -R -p1 < "$PATCH" ) >/dev/null 2>&1 || true
  # the cycle leaves the RELINKED candidate .so in the build tree; restore the
  # proven pristine champion bytes (md5 2361042a185a7562c6ba5087eeaa89a0) and
  # recompile the build-dir mmvq.cpp.o from champion source so the tree is
  # self-consistent for the next cycle (lmhead-q8 LEDGER: done manually post-
  # cycle; this harness automates it).
  if [ -f "$CHAMP_SO" ]; then
    cp "$CHAMP_SO" "$BUILD/bin/libggml-sycl.so.0.17.0"
    MMVQ_OBJ=$(find "$BUILD" -name 'mmvq.cpp.o' | head -1)
    if [ -n "$MMVQ_OBJ" ] && [ -z "${SKIP_RESTORE_MMVQ:-}" ] && [ -n "${ICPC:-}" ]; then
      "$ICPC" "${DEFS[@]}" "${INCS[@]}" -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-unused-function -Wno-narrowing -fsycl -DMKL_ILP64 \
        -o "$MMVQ_OBJ" -c "$SRC/ggml/src/ggml-sycl/mmvq.cpp" >/dev/null 2>&1 || true
    fi
    echo "[$LABEL] champion .so restored: $(md5sum "$BUILD/bin/libggml-sycl.so.0.17.0" | cut -c1-16)..."
  else
    echo "[$LABEL] WARN: no champion .so at $CHAMP_SO — candidate .so left in build tree"
  fi
}
trap 'restore_champion' EXIT
echo "[$LABEL] applied $PATCH"

# --- compile probe objects (BOTH TUs), swap, relink --------------------------
[ -f "$BUILD/probe-build.sh" ] || { echo "missing $BUILD/probe-build.sh"; exit 2; }
[ -x "$BUILD/bin/llama-bench" ] || { echo "missing champion llama-bench"; exit 2; }
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:/opt/intel/oneapi/compiler/2026.0/bin:${LD_LIBRARY_PATH:-}
if [ -z "${SKIP_COMPILE:-}" ] || [ ! -f "$BUILD/probe-ggml-sycl.o" ]; then
  bash "$BUILD/probe-build.sh" >/tmp/$LABEL-build.log 2>&1 \
    || { echo "BUILD FAIL:"; tail -20 /tmp/$LABEL-build.log; exit 3; }
fi
ICPC=/opt/intel/oneapi/compiler/2026.0/bin/icpx
DEFS=(-DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_SYCL_DNNL=1 -DGGML_SYCL_F16 -DGGML_SYCL_GRAPH -DGGML_SYCL_HOST_MEM_FALLBACK -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API -DGGML_SYCL_WARP_SIZE=16 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_sycl_EXPORTS)
INCS=(-I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/ggml-sycl/.. -I/home/frosty40/turbo/worktrees/treebeard-base-control-latest/ggml/src/../include -isystem /opt/intel/oneapi/compiler/2026.0/include -isystem /opt/intel/oneapi/dnnl/2026.0/include -isystem /opt/intel/oneapi/mkl/2026.0/include)
if [ -z "${SKIP_COMPILE:-}" ] || [ ! -f "$BUILD/probe-mmvq.o" ]; then
  "$ICPC" "${DEFS[@]}" "${INCS[@]}" -O3 -DNDEBUG -std=gnu++17 -fPIC -Wno-unused-function -Wno-narrowing -fsycl -DMKL_ILP64 \
    -o "$BUILD/probe-mmvq.o" -c "$SRC/ggml/src/ggml-sycl/mmvq.cpp" >>/tmp/$LABEL-build.log 2>&1 \
    || { echo "MMVQ BUILD FAIL:"; tail -30 /tmp/$LABEL-build.log; exit 4; }
fi
OBJDIR=$(dirname "$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)")
MMVQ_OBJ=$(find "$BUILD" -name 'mmvq.cpp.o' | head -1)
[ -n "$OBJDIR" ] && [ -n "$MMVQ_OBJ" ] || { echo "[$LABEL] missing build objects"; exit 5; }
cp "$BUILD/probe-ggml-sycl.o" "$OBJDIR/ggml-sycl.cpp.o"
cp "$BUILD/probe-mmvq.o" "$MMVQ_OBJ"
echo "[$LABEL] swapped ggml-sycl.cpp.o + mmvq.cpp.o"
LINK="$OBJDIR/link.txt"
if [ -f "$LINK" ]; then
  ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >/tmp/$LABEL-link.log 2>&1 \
    || { echo "RELINK FAIL:"; tail -20 /tmp/$LABEL-link.log; exit 5; }
else
  cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/$LABEL-link.log 2>&1 \
    || { echo "CMAKE RELINK FAIL:"; tail -20 /tmp/$LABEL-link.log; exit 5; }
fi
echo "[$LABEL] probe built, swapped, relinked"

# --- golden-smoke WITH the gate ON (quality-invisibility gate) ---------------
if [ "${GOLDEN:-1}" = "1" ]; then
  GATE="$GATE" bash scripts/golden-smoke.sh >/tmp/$LABEL-golden.log 2>&1 \
    || { echo "GOLDEN FAIL (gate not quality-neutral):"; tail -20 /tmp/$LABEL-golden.log; exit 6; }
  echo "[$LABEL] GOLDEN OK"
fi

if [ "${BENCH:-0}" != "1" ]; then
  echo "[$LABEL] probe built + golden OK. BENCH=1 to run the same-window 3-arm A/B."
  exit 0
fi

# --- same-window CTRL vs candidate sandwich (official geometry) --------------
echo "[$LABEL] arm1 CTRL (gate off) -> arm2 candidate (gate on) -> arm3 CTRL"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT-$STAMP"
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
./scripts/with-gpu-lock --wait -- env -u "$GATE" "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m "$MODEL" \
  > "$OUT-$STAMP/ctrl-a.log" 2>&1 || { echo "CTRL-A FAIL"; exit 7; }
./scripts/with-gpu-lock --wait -- env "$GATE"=1 "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m "$MODEL" \
  > "$OUT-$STAMP/cand.log" 2>&1 || { echo "CAND FAIL"; exit 7; }
./scripts/with-gpu-lock --wait -- env -u "$GATE" "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m "$MODEL" \
  > "$OUT-$STAMP/ctrl-b.log" 2>&1 || { echo "CTRL-B FAIL"; exit 7; }

echo "[$LABEL] A/B window complete: $OUT-$STAMP/"
echo "[$LABEL] next: parse the three logs (per findings 18/40, JSONDecoder-scan, strip [lx-*])"
exit 0
