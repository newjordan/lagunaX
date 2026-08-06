#!/bin/bash
# lmhead-probe-cycle.sh — build the env-gated lm_head timer probe into
# src-lmhead-build and measure the per-token lm_head share of the decode
# budget on the B70 card (FRONTIER_20260729 idea #7: candidate-table prune).
#
# Probe instrumentation in src-lmhead/ggml/src/ggml-sycl/ggml-sycl.cpp
# (env-gated: GGML_SYCL_LMHEAD_TIMER / GGML_SYCL_SKIP_LMHEAD / GGML_SYCL_LMHEAD_LAYER)
# is pure chrono around the fused l_out-<L> GEMV dispatch; it must be linked
# into the REAL champion binary (src-lmhead-build) to measure the shipped path.
#
# Sequence: golden-smoke WITH timer ON (proves instrumentation is invisible to
# output) -> timed official bench (pp512/tg128) -> skip-mode diagnostic bench
# (NOT golden-gated: it intentionally corrupts output; it bounds the ceiling
# if the lm_head GEMV were free).
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

[ -f "$BUILD/probe-build.sh" ] || { echo "missing $BUILD/probe-build.sh"; exit 2; }
[ -x "$BUILD/bin/llama-bench" ] || { echo "missing champion llama-bench"; exit 2; }

echo "[lmhead-probe] (1/5) compiling ggml-sycl.cpp -> probe object"
if [ -z "${SKIP_COMPILE:-}" ] || [ ! -f "$BUILD/probe-ggml-sycl.o" ]; then
  bash "$BUILD/probe-build.sh" >/tmp/lmhead-probe-build.log 2>&1 \
    || { echo "BUILD FAIL:"; tail -20 /tmp/lmhead-probe-build.log; exit 3; }
else
  echo "[lmhead-probe] SKIP_COMPILE set — reusing existing probe object"
fi
[ -f "$BUILD/probe-ggml-sycl.o" ] || { echo "probe object not produced"; exit 3; }

echo "[lmhead-probe] (2/5) swapping probe object into build tree"
OBJ=$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)
if [ -z "$OBJ" ]; then
  # Makefile-style build may name the TU object differently; fall back to a
  # full cmake rebuild of the target (which recompiles from src-lmhead source,
  # probe code included).
  touch "$SRC/ggml/src/ggml-sycl/ggml-sycl.cpp"
  cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/lmhead-probe-link.log 2>&1 \
    || { echo "CMAKE BUILD FAIL:"; tail -20 /tmp/lmhead-probe-link.log; exit 4; }
  echo "[lmhead-probe] cmake rebuilt ggml-sycl (no cpp.o found for swap)"
else
  cp "$BUILD/probe-ggml-sycl.o" "$OBJ"
  LINK=$(dirname "$OBJ")/link.txt
  if [ -f "$LINK" ]; then
    ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >/tmp/lmhead-probe-link.log 2>&1 \
      || { echo "RELINK FAIL:"; tail -20 /tmp/lmhead-probe-link.log; exit 5; }
    echo "[lmhead-probe] relinked ggml-sycl via $(basename "$(dirname "$LINK")")/link.txt"
  else
    cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/lmhead-probe-link.log 2>&1 \
      || { echo "CMAKE RELINK FAIL:"; tail -20 /tmp/lmhead-probe-link.log; exit 5; }
  fi
fi

MODEL=${MODEL:-}
[ -f ./env.sh ] && source ./env.sh >/dev/null 2>&1 || true
if [ -z "$MODEL" ]; then
  MODEL=${MODEL:-$(find "$ROOT/baseline" "$ROOT" -maxdepth 3 -name '*.gguf' 2>/dev/null | head -1)}
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "no model found (set MODEL=... and pass it)"; exit 6; }
echo "[lmhead-probe] model: $MODEL"

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/results/lmhead-probe-$TS
mkdir -p "$OUT"

echo "[lmhead-probe] (3/5) golden-smoke with timer ON (instrumentation must be invisible)"
if [ -x "$ROOT/scripts/golden-smoke.sh" ]; then
  GGML_SYCL_LMHEAD_TIMER=1 bash "$ROOT/scripts/golden-smoke.sh" >"$OUT/golden-timer.log" 2>&1 \
    || { echo "GOLDEN FAIL with timer ON:"; tail -10 "$OUT/golden-timer.log"; exit 7; }
  echo "[lmhead-probe] golden OK with timer instrumentation live"
fi

echo "[lmhead-probe] (4/5) timed official bench (pp512/tg128, timer ON)"
BENCH_ARGS="-m $MODEL -p 512 -n 128"
export GGML_SYCL_DISABLE_GRAPH=1
export LD_LIBRARY_PATH="$BUILD/bin:${LD_LIBRARY_PATH:-}"
echo "[lmhead-probe] probed lib: $(ldd "$BUILD/bin/llama-bench" | grep -m1 ggml-sycl | awk '{print $3}')"
GGML_SYCL_LMHEAD_TIMER=1 "$BUILD/bin/llama-bench" $BENCH_ARGS \
  >"$OUT/bench-timer.log" 2>"$OUT/bench-timer.stderr" || { echo "TIMED BENCH FAIL"; exit 8; }
grep -m1 'FINAL' "$OUT/bench-timer.stderr" | tee "$OUT/lmhead-final.txt"
if ! grep -q 'FINAL' "$OUT/lmhead-final.txt"; then
  echo "NO TIMER LINES — probe not linked into this binary; aborting"
  exit 9
fi

echo "[lmhead-probe] (5/5) SKIP-mode diagnostic bench (output-corrupting; ceiling bound only)"
set +e
GGML_SYCL_SKIP_LMHEAD=1 GGML_SYCL_LMHEAD_TIMER=1 "$BUILD/bin/llama-bench" $BENCH_ARGS \
  >"$OUT/bench-skip.log" 2>"$OUT/bench-skip.stderr"
SKIP_RC=$?
set -e
echo "[lmhead-probe] skip bench rc=$SKIP_RC (nonzero expected: garbage output)"
grep -m1 'FINAL' "$OUT/bench-skip.stderr" | tee "$OUT/lmhead-skip-final.txt" || true

echo "[lmhead-probe] DONE -> $OUT"
echo "timed:"
grep -E 'pp512|tg128' "$OUT/bench-timer.log" | tail -2
echo "skip (ceiling):"
grep -E 'pp512|tg128' "$OUT/bench-skip.log" | tail -2
