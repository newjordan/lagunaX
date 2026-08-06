#!/usr/bin/env bash
# prefill-budget.sh — per-bucket GPU dispatch budget for a PREFILL-heavy run.
#
# Why: the LayerTimer probe (finding 23) was measured on decode-only runs
# (tg128/tg256). Prefill is the highest-unknown axis (1.018x vs the 2.0x
# target; decode levers exhausted per findings 8-13/26-28/34) and NO
# source-level prefill budget has ever been captured. This cycle:
#   1. verifies the installed champion .so still carries the layer-timer
#      instrumentation (4 strings); if not, rebuilds the probe object via the
#      exact compile_commands.json flags (NOT cmake --build, which no-ops on
#      the write-denied tree), relinks via CMake link.txt, and installs it
#      into the champion bin tree — same proven loop as layer-timer-cycle.sh.
#   2. runs llama-bench at the official prefill geometry (pp512) with
#      GGML_SYCL_TIMER_ALL=1 so the fused dispatch buckets are attributed to
#      a prefill-dominated run.
#
# Load-bearing env facts (same as layer-timer-cycle.sh):
#   * LD_LIBRARY_PATH must PREPEND (clobber -> rc=134 "No device of
#     requested type": the L0 adapter needs libumf.so.1 from umf/1.1/lib).
#   * ONEAPI_DEVICE_SELECTOR=level_zero:gpu, ZE_AFFINITY_MASK=0.
#   * The bench build has no -tb / --sycl-disable-* flags (env-only gates).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src-lmhead/ggml/src/ggml-sycl/ggml-sycl.cpp"
BUILD="$ROOT/src-lmhead-build"
BINTREE="$ROOT/results/src-repro-20260806T035656Z/bin"
MODEL="/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf"
OUT="${1:-$ROOT/results/prefill-budget-$(date +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:$BINTREE:${LD_LIBRARY_PATH:-}"

HAVE_TIMER=$(strings "$BINTREE/libggml-sycl.so.0.17.0" 2>/dev/null | grep -c 'layer-timer' || true)
echo "timer strings in installed .so: $HAVE_TIMER" | tee "$OUT/so-check.log"

if [ "${HAVE_TIMER:-0}" -lt 2 ]; then
  echo ">> rebuilding probe object (timer missing from installed .so)" | tee -a "$OUT/so-check.log"
  CMD=$(python3 - "$BUILD" "$SRC" <<'EOF'
import json, sys
build, src = sys.argv[1], sys.argv[2]
cc = json.load(open(build + '/compile_commands.json'))
for e in cc:
    if e['file'].endswith('ggml-sycl.cpp'):
        c = e['command']
        i = c.rfind(' -c ')
        print(c[:i] + ' -c ' + src)
        break
EOF
)
  cd "$BUILD/ggml/src/ggml-sycl" || exit 1
  bash -c "$CMD" >>"$OUT/compile.log" 2>&1 || { echo "COMPILE FAIL"; exit 2; }
  bash CMakeFiles/ggml-sycl.dir/link.txt >>"$OUT/compile.log" 2>&1 || { echo "LINK FAIL"; exit 2; }
  cp -f --remove-destination ../../../bin/libggml-sycl.so.0.17.0 "$BINTREE/libggml-sycl.so.0.17.0"
  echo ">> installed $(strings "$BINTREE/libggml-sycl.so.0.17.0" | grep -c layer-timer) timer strings" | tee -a "$OUT/so-check.log"
fi

# 2) prefill-dominated official-geometry bench with the timer live.
#    PP=512 matches the official prefill shape; NN=16 keeps tg noise tiny so
#    the bucket totals are prefill-attributed. r=2 for a stable mean.
#    NOTE: run DIRECTLY (not via with-gpu-lock) — the lock wrapper sources
#    env.sh whose line 89 prepends $LX_BIN (timer-free mmadd-decode build)
#    AHEAD of $BINTREE, so the instrumented .so is silently shadowed and the
#    timer prints nothing. Direct invocation keeps $BINTREE first.
cd "$ROOT" || exit 1
GGML_SYCL_TIMER_ALL=1 "$BINTREE/llama-bench" -m "$MODEL" -p 512 -n 16 \
  -t 16 -ub 2048 -b 2048 -ngl 99 -r 2 -ctk f16 -ctv f16 \
  >"$OUT/bench.log" 2>"$OUT/bench.stderr"
echo "rc=$?" | tee "$OUT/rc.txt"
grep -E 'layer-timer|lmhead-probe' "$OUT/bench.stderr" | tee "$OUT/budget.txt"
echo "--- per-bucket rows (pp512-attributed) ---"
grep -E '^\[layer-timer\] bucket' "$OUT/budget.txt" | tee "$OUT/buckets.txt"
