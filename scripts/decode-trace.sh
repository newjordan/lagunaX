#!/usr/bin/env bash
# Decode kernel trace of the champion binary via oneAPI sycl-trace.
# Measures the actual per-kernel decode timeline (lm_head GEMV share, launch gaps).
set -u
LX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${LX_BIN:-$LX_ROOT/results/src-repro-20260806T035656Z/bin/llama-bench}"
MODEL="${LX_MODEL:-/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf}"
OUT="${1:-$LX_ROOT/results/decode-trace-$(date +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-1}"
export GGML_SYCL_DISABLE_DNN="${GGML_SYCL_DISABLE_DNN:-1}"
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:$(dirname "$BIN"):${LD_LIBRARY_PATH:-}"

TRACE_TOOL=/opt/intel/oneapi/compiler/latest/bin/sycl-trace
ARGS=(-m "$MODEL" -ngl 99 -t 16 -ub 2048 -b 2048 -p "${PP:-16}" -n "${NN:-64}" -r 1 -ctk f16 -ctv f16)

echo ">> binary: $BIN" | tee "$OUT/header.txt"
echo ">> args: ${ARGS[*]}" | tee -a "$OUT/header.txt"
echo ">> trace: level_zero" | tee -a "$OUT/header.txt"

"$TRACE_TOOL" --level_zero --print-format=compact -- "$BIN" "${ARGS[@]}" > "$OUT/trace.txt" 2>&1
rc=$?
echo "rc=$rc" | tee -a "$OUT/header.txt"

python3 - "$OUT/trace.txt" > "$OUT/summary.txt" 2>&1 <<'EOF'
import sys, re, collections
path = sys.argv[1]
ev = collections.defaultdict(list)
cur = None
pat_k = re.compile(r"kernel\s*=\s*(\S+)")
pat_t = re.compile(r"time\s*=\s*([0-9.]+)\s*ms")
with open(path, errors="replace") as f:
    lines = f.readlines()
for line in lines:
    m = pat_k.search(line)
    if m:
        cur = m.group(1)
    t = pat_t.search(line)
    if t and cur is not None:
        ev[cur].append(float(t.group(1)))
tot = sum(sum(v) for v in ev.values())
print(f"trace lines: {len(lines)}  distinct kernels: {len(ev)}  total ms: {tot:.2f}")
for k, v in sorted(ev.items(), key=lambda kv: -sum(kv[1]))[:25]:
    print(f"{sum(v):9.2f} ms  calls={len(v):5d}  avg={sum(v)/len(v)*1000:7.1f} us  {k}")
EOF
echo "=== summary ==="
cat "$OUT/summary.txt"
echo "OUT=$OUT"
