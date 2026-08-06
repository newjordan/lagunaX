#!/usr/bin/env bash
# lmhead-timer-probe.sh — source-level lm_head measurement probe (open-lead 19/24).
#
# Why this exists: the previous iteration's blocker was that the champion source worktree
# (/home/frosty40/turbo/worktrees/treebeard-base-control-latest) is READ-ONLY to this sandbox,
# and sycl-trace cannot emit per-kernel timings (finding 18). This script stages a WRITABLE
# copy under $LX_ROOT/src-lmhead and patches ggml-sycl.cpp with an ENV-GATED, self-locating
# instrumentation block (no-op unless GGML_SYCL_LMHEAD_TIMER=1 / GGML_SYCL_SKIP_LMHEAD=1),
# then builds with the pinned icpx/icx toolchain mirroring the src-repro CMakeCache.
#
# Two measurements, both anchored at ggml_sycl_mul_mat on the fused final-layer group
# (mm='ffn_shexp-39' add='ffn_out-39' add2='l_out-39', per finding 16):
#   A) TIMER=1        — count + inter-call gap of l_out dispatches (per-token cycle view).
#   B) SKIP_LMHEAD=1  — replace the lm_head GEMV with a fill of the small logits dst
#                       (ggml_backend_tensor_set, public API; the ~168 MB weight stream is
#                       never read). tg delta vs (A) = exact lm_head GEMV share of decode.
# The ablation deliberately produces garbage logits, so it is gated OUT of golden-smoke:
# (A) must pass golden; (B) is a labeled diagnostic only.
#
# Usage:
#   bash scripts/lmhead-timer-probe.sh stage   # writable copy + apply patch (validated)
#   bash scripts/lmhead-timer-probe.sh build   # cmake + build llama-bench (mirror src-repro)
#   bash scripts/lmhead-timer-probe.sh probe   # golden-gated timed run + ablation, one lock window
#   bash scripts/lmhead-timer-probe.sh all     # stage→build→probe
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

SRC_ORIG="${LX_SRC_SYCL:-/home/frosty40/turbo/worktrees/treebeard-base-control-latest}"
STAGE="$ROOT/src-lmhead"
SYCL_CPP="$STAGE/ggml/src/ggml-sycl/ggml-sycl.cpp"
BIN="$STAGE/build/bin/llama-bench"

stage() {
  if [[ ! -d "$SRC_ORIG/ggml/src/ggml-sycl" ]]; then
    echo "source tree missing: $SRC_ORIG" >&2; exit 1
  fi
  mkdir -p "$ROOT/src-lmhead"
  if [[ ! -f "$SYCL_CPP" ]]; then
    echo "staging writable copy of $SRC_ORIG ..."
    cp -a "$SRC_ORIG/." "$STAGE/"
  fi
  [[ -f "$SYCL_CPP" ]] || { echo "stage failed" >&2; exit 1; }
  python3 - "$SYCL_CPP" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()

if "LmheadProbe" in src:
    print("probe patch already present — skipping")
    sys.exit(0)

fn = "ggml_sycl_mul_mat("
i = src.find(fn)
if i < 0:
    print(f"ANCHOR MISSING: {fn} not found in {p}", file=sys.stderr)
    sys.exit(2)
brace = src.find("{", i)
if brace < 0 or brace - i > 4000:
    print("ANCHOR MISSING: no opening brace near ggml_sycl_mul_mat", file=sys.stderr)
    sys.exit(2)
insert_at = brace + 1

block = '''
    // [LmheadProbe] env-gated source-level lm_head measurement. No-op in normal runs:
    // both branches require their GGML_SYCL_* env vars, so the default binary is
    // behaviorally identical (golden-safe). Diagnostic only; never part of scored runs.
    {
        static struct LmheadProbeState {
            std::chrono::steady_clock::time_point last{};
            uint64_t n = 0, us = 0;
            void * zeros = nullptr; size_t zeros_sz = 0;
        } st;
        const bool is_lmhead = dst && dst->name
            && (strstr(dst->name, "l_out") || strstr(dst->name, "result_output"));
        if (is_lmhead) {
            const bool timer = getenv("GGML_SYCL_LMHEAD_TIMER") != nullptr;
            const bool skip  = getenv("GGML_SYCL_SKIP_LMHEAD") != nullptr;
            if (timer) {
                auto _now = std::chrono::steady_clock::now();
                if (st.last.time_since_epoch().count()) {
                    st.us += (uint64_t) std::chrono::duration_cast<
                        std::chrono::microseconds>(_now - st.last).count();
                }
                st.last = _now;
                st.n++;
            }
            if (skip) {
                // Replace the lm_head GEMV (weight stream ~168 MB/token, never read here)
                // with a fill of the small logits dst via the public backend API.
                const size_t nb = ggml_nbytes(dst);
                if (!st.zeros) { st.zeros = calloc(1, nb); st.zeros_sz = nb; }
                if (st.zeros_sz < nb) { st.zeros = realloc(st.zeros, nb); st.zeros_sz = nb; memset(st.zeros, 0, nb); }
                ggml_backend_tensor_set(dst, st.zeros, 0, nb);
                if (timer) {
                    fprintf(stderr, "[lmhead-probe] skipped lm_head GEMV (dst=%s, %zu bytes)\\n",
                            dst->name ? dst->name : "?", nb);
                }
                return;
            }
            if (timer && st.n && (st.n % 64) == 0) {
                fprintf(stderr, "[lmhead-probe] %llu l_out dispatches, %llu us so far, %.2f us/cycle\\n",
                        (unsigned long long) st.n, (unsigned long long) st.us,
                        (double) st.us / (double) st.n);
            }
        }
    }
'''
src = src[:insert_at] + block + src[insert_at:]
if "#include <chrono>" not in src:
    src = src.replace("#include <memory>", "#include <memory>\n#include <chrono>", 1)
open(p, "w").write(src)
print(f"probe patch applied at {p}:{insert_at}")
PY
  touch "$STAGE/.lmhead-probe-patched"
  echo "stage OK — patched source at $SYCL_CPP"
}

build() {
  [[ -f "$SYCL_CPP" ]] || { echo "run stage first" >&2; exit 1; }
  local bd="$STAGE/build"
  if [[ ! -f "$bd/CMakeCache.txt" ]]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    cmake -S "$STAGE" -B "$bd" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icx \
      -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icpx \
      -DGGML_SYCL=ON -DGGML_SYCL_DNN=ON \
      -DGGML_LLAMA_API=ON -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF
  fi
  source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
  cmake --build "$bd" --target llama-bench -j "$(nproc)"
  [[ -x "$BIN" ]] || { echo "llama-bench missing after build" >&2; exit 1; }
  echo "build OK — $BIN"
}

bench_cmd() {
  echo "$BIN" -m "$LX_MODEL" -ngl "$NGL" \
    --n-cpu-moe "$LX_CPU_MOE_LAYERS" --split-mode "$LX_SPLIT_MODE" \
    --main-gpu "$LX_MAIN_GPU" --tensor-split "$LX_TENSOR_SPLIT" --device "$LX_DEVICE" \
    -t "$THREADS" --cpu-mask "$LX_CPU_MASK" --cpu-strict "$LX_CPU_STRICT" \
    -b "$BBATCH" -ub "$UBATCH" -ctk "$CTK" -ctv "$CTV" \
    --no-kv-offload "$LX_NO_KV_OFFLOAD" --no-op-offload "$LX_NO_OP_OFFLOAD" --no-host "$LX_NO_HOST" \
    -r "$LX_REPS" -d "$LX_DEPTH" --prio "$LX_PRIO" --load-mode "$LX_LOAD_MODE" \
    --poll "$LX_POLL" -p "$LX_PPROMPT" -n "$LX_TGEN"
}

probe() {
  [[ -x "$BIN" ]] || { echo "run build first" >&2; exit 1; }
  lx_gpu_lock_enter "lmhead-timer-probe" || exit $?
  trap 'lx_gpu_lock_leave' EXIT
  source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
  export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:/opt/intel/oneapi/compiler/2026.0/lib/linux:$LD_LIBRARY_PATH"

  # Pass A: golden-gated timed run (normal path, timer on).
  bash "$ROOT/scripts/golden-smoke.sh" || { echo "GOLDEN FAIL — abort probe" >&2; exit 1; }
  local stamp_a; stamp_a="$(date -u +%Y%m%dT%H%M%SZ)"
  local out_a="$ROOT/results/lmhead-timer-$stamp_a"
  mkdir -p "$out_a"
  GGML_SYCL_LMHEAD_TIMER=1 $(bench_cmd) 2>"$out_a/probe.log" | tee "$out_a/llama-bench.log"
  echo "pass A (timer) output:"; grep '\[lmhead-probe\]' "$out_a/probe.log" || echo "NO PROBE LINE"

  # Pass B: ablation (skip lm_head GEMV). Diagnostic only — golden would fail by design.
  local stamp_b; stamp_b="$(date -u +%Y%m%dT%H%M%SZ)"
  local out_b="$ROOT/results/lmhead-skip-$stamp_b"
  mkdir -p "$out_b"
  GGML_SYCL_LMHEAD_TIMER=1 GGML_SYCL_SKIP_LMHEAD=1 $(bench_cmd) 2>"$out_b/probe.log" | tee "$out_b/llama-bench.log"
  echo "pass B (skip) output:"; grep '\[lmhead-probe\]' "$out_b/probe.log" || echo "NO PROBE LINE"
  echo "results in $out_a and $out_b"
}

case "${1:-all}" in
  stage) stage ;;
  build) build ;;
  probe) probe ;;
  all) stage; build; probe ;;
  *) echo "usage: $0 [stage|build|probe|all]" >&2; exit 2 ;;
esac
