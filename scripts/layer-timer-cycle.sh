#!/usr/bin/env bash
# layer-timer-cycle.sh — instrumented-champion rebuild + per-layer decode budget.
#
# Why: sycl-trace is dead on this stack (findings 18), so per-kernel budgets
# come from an in-source env-gated chrono probe (GGML_SYCL_TIMER_ALL=1) at the
# fused mul_mat dispatch site in ggml-sycl.cpp. This cycle rebuilds ONLY the
# changed object via the exact compile_commands flags (NOT cmake --build, which
# tracks the write-denied base-control source and silently no-ops), relinks the
# shared lib via the object dir's link.txt, installs it into the champion
# binary tree, and runs one decode-only bench with the timer live.
#
# Env facts (load-bearing):
#   * LD_LIBRARY_PATH must be PREPENDED, never clobbered: the Level Zero UR
#     adapter needs libumf.so.1 from /opt/intel/oneapi/umf/1.1/lib, which
#     setvars.sh adds. A clobbering export drops it -> rc=134 "No device of
#     requested type available" (fake device failure, real loader failure).
#   * ONEAPI_DEVICE_SELECTOR=level_zero:gpu and ZE_AFFINITY_MASK=0 required.
#   * cmake --build --target ggml-sycl is a silent no-op: deps point at
#     /home/frosty40/turbo/worktrees/treebeard-base-control-latest (write-
#     denied, mtime unchanged). The probe cycle compiles the edited source
#     directly with the compile_commands.json command and swaps the .o.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src-lmhead/ggml/src/ggml-sycl/ggml-sycl.cpp"
BUILD="$ROOT/src-lmhead-build"
BINTREE="$ROOT/results/src-repro-20260806T035656Z/bin"
MODEL="/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf"
OUT="${1:-$ROOT/results/layer-timer/run-$(date +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:$BINTREE:${LD_LIBRARY_PATH:-}"

# 1) rebuild the edited object with the exact configure-time flags
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
echo ">> compiling $SRC" | tee "$OUT/compile.log"
bash -c "$CMD" >>"$OUT/compile.log" 2>&1 || { echo "COMPILE FAIL"; exit 2; }
echo ">> relinking libggml-sycl.so" | tee -a "$OUT/compile.log"
bash CMakeFiles/ggml-sycl.dir/link.txt >>"$OUT/compile.log" 2>&1 || { echo "LINK FAIL"; exit 2; }
cp -f --remove-destination ../../../bin/libggml-sycl.so.0.17.0 "$BINTREE/libggml-sycl.so.0.17.0"
echo ">> installed $BINTREE/libggml-sycl.so.0.17.0"

# 2) decode-only bench with the timer live
GGML_SYCL_TIMER_ALL=1 "$BINTREE/llama-bench" -m "$MODEL" -p "${PP:-4}" -n "${NN:-128}" \
  -t 16 -ub 2048 -b 2048 -ngl 99 -r 1 -ctk f16 -ctv f16 \
  >"$OUT/bench.log" 2>"$OUT/bench.stderr"
echo "rc=$?" | tee "$OUT/rc.txt"
grep -E 'layer-timer|lmhead-probe' "$OUT/bench.stderr" | tee "$OUT/budget.txt"
