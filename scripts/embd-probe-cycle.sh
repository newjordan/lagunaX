#!/usr/bin/env bash
# embd-probe-cycle.sh — batch-512 embedding (get_rows/tok_embd) dispatch-budget probe.
#
# The layer-timer bucket classifier has a tok_embd bucket (bucket 4) that NEVER fires:
# the batch-512 embedding gather runs through ggml_get_rows, which is not instrumented
# (results/layer-timer/LEDGER.md, bucket-instrumentation audit). This harness wires the
# get_rows dispatch into the same chrono bucket and runs the full gate chain:
#
#   probe build -> golden smoke -> official-geometry timed bench (direct llama-bench
#   invocation, NOT via with-gpu-lock, to avoid the LX_BIN lib-shadowing bug class,
#   finding 33/34) -> probe revert + champion .so restore (finding 27 hygiene).
#
# Usage: bash scripts/embd-probe-cycle.sh [patch-file] [bench|probe|all]
#   patch-file default: patches/embd-bucket-timer.patch
#   mode default: all (build+golden+bench); "bench" skips rebuild when probe is live.
set -uo pipefail
LX=${LX:-/home/frosty40/turbo/lx}
PATCH=${1:-$LX/patches/embd-bucket-timer.patch}
# make absolute so git apply -R works from the src-lmhead cwd
case "$PATCH" in /*) ;; *) PATCH="$LX/$PATCH" ;; esac
MODE=${2:-all}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$LX/results/embd-probe-$STAMP"
mkdir -p "$OUT"
exec > >(tee "$OUT/cycle.log") 2>&1

SRC="$LX/src-lmhead/ggml/src/ggml-sycl"
BUILD="$LX/src-lmhead-build"
# champion .so backup (finding 23: md5 2361042a185a7562c6ba5087eeaa89a0)
EMB_BIN="$LX/results/src-repro-20260806T035656Z/bin"
CHAMP_SO="$EMB_BIN/libggml-sycl.so.0.17.0"
BK_SO="$LX/results/src-repro-20260806T035656Z/bin/libggml-sycl.so.0.17.0.orig"

# --- 0. sanity ---------------------------------------------------------------
[ -f "$PATCH" ] || { echo "FATAL: patch $PATCH missing"; exit 2; }
[ -f "$SRC/ggml-sycl.cpp" ] || { echo "FATAL: source tree missing at $SRC"; exit 2; }
# never clobber LD_LIBRARY_PATH — PREPEND umf (finding 15)
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
command -v icpx >/dev/null || { echo "FATAL: icpx not on PATH (need setvars)"; exit 2; }
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"

# --- 1. apply patch to source (patch --forward, revert via git apply -R, finding 27)
cd "$LX/src-lmhead" || exit 2
if [ "$MODE" != "bench" ]; then
  if grep -q "GetRowsTimer" "$SRC/ggml-sycl.cpp"; then
    echo "FATAL: gate already present — revert with: git -C $LX/src-lmhead apply -R '$PATCH'"; exit 3
  fi
  patch --forward -p1 < "$PATCH" > "$OUT/patch.log" 2>&1
  rc=$?
  echo "patch rc=$rc, timer marker present: $(grep -c GetRowsTimer "$SRC/ggml-sycl.cpp" || true)"
  [ $rc -eq 0 ] || { echo "FATAL: patch failed rc=$rc"; exit 4; }
fi

# --- 2. build probe object, swap, relink (mirrors lmhead-prefetch-cycle.sh) ---
if [ "$MODE" != "bench" ]; then
  cd "$BUILD" || exit 2
  cp "$CHAMP_SO" "$BK_SO" 2>/dev/null || true
  bash probe-build.sh > "$OUT/build.log" 2>&1
  grep -q "PROBE_OBJECT_BUILT" "$OUT/build.log" || { echo "FATAL: probe build failed"; git -C "$LX/src-lmhead" apply -R "$PATCH" 2>/dev/null; exit 5; }
  OBJ=$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)
  [ -n "$OBJ" ] || { echo "FATAL: no ggml-sycl.cpp.o in build tree"; exit 5; }
  cp -f "$BUILD/probe-ggml-sycl.o" "$OBJ"
  LINK="$(dirname "$OBJ")/link.txt"
  if [ -f "$LINK" ]; then
    ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >> "$OUT/build.log" 2>&1 \
      || { echo "FATAL: relink failed"; git -C "$LX/src-lmhead" apply -R "$PATCH" 2>/dev/null; cp "$BK_SO" "$CHAMP_SO"; exit 5; }
  else
    echo "FATAL: no link.txt at $LINK"; exit 5
  fi
  # install freshly relinked lib into the champion binary tree
  NEWSO=$(find "$BUILD" -name 'libggml-sycl.so*' -newer "$BK_SO" | head -1)
  if [ -n "$NEWSO" ]; then
    cp -f "$NEWSO" "$CHAMP_SO"
    echo "probe lib installed: $(md5sum "$CHAMP_SO" | cut -c1-16)"
  else
    echo "FATAL: relinked lib not found"; exit 5
  fi
fi

# --- 3. golden smoke gate (MUST see the probe lib, not LX_BIN timer-free lib) --
cd "$LX" || exit 2
source "$LX/env.sh" >/dev/null 2>&1
export LD_LIBRARY_PATH="$EMB_BIN:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
LX_LLAMA_SERVER="$EMB_BIN/llama-server" \
  bash scripts/golden-smoke.sh > "$OUT/golden.log" 2>&1
grep -qi "golden ok" "$OUT/golden.log" || { echo "FATAL: golden failed"; git -C "$LX/src-lmhead" apply -R "$PATCH" 2>/dev/null; cp "$BK_SO" "$CHAMP_SO"; exit 6; }
echo "GOLDEN OK"

# --- 4. official-geometry timed bench, DIRECT invocation (finding 33/34) ------
# geometry pinned in results/20260806T060605Z/metrics.json (finding 22)
if [ "$MODE" != "probe" ]; then
  source "$LX/env.sh" >/dev/null 2>&1
  # env.sh prepends $LX_BIN (timer-free mmadd-decode build) AHEAD of $BINTREE
  # (finding 33/34 bug class) — re-prepend the probe bin dir so the timer fires.
  export LD_LIBRARY_PATH="$EMB_BIN:${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
  export GGML_SYCL_TIMER_ALL=1
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 \
         GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1
  BENCH="$LX/results/src-repro-20260806T035656Z/bin/llama-bench"
  MODEL="${LX_MODEL:-/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf}"
  "$BENCH" -m "$MODEL" -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto \
    -b 2048 -ub 2048 -ctk f16 -ctv f16 -r 5 -o json \
    -p 512 -n 0 > "$OUT/bench.json" 2> "$OUT/bench.stderr"
  echo "BENCH rc=$?"
fi

# --- 5. revert probe + restore champion .so (finding 27 hygiene) --------------
git -C "$LX/src-lmhead" apply -R "$PATCH" > /dev/null 2>&1
cp "$BK_SO" "$CHAMP_SO"
echo "probe reverted; champion .so restored"
echo "CYCLE_STAMP=$STAMP"
echo "BUCKET_FILE=$OUT/bench.stderr"
