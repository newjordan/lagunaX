#!/usr/bin/env bash
# LmHeadProbe: measure the lm_head fused-group time share on the B70 via a
# source-level env-gated probe binary (libggml-sycl.so built from the patched
# source with the EXACT champion compile flags). Diagnostic only — never a
# scored run. Sequence under ONE gpu-lock window:
#   1) golden-smoke on the probe binary (default env => inert)  [quality gate]
#   2) timer run:  pp512/tg128 combined, GGML_SYCL_LMHEAD_TIMER=1
#   3) skip run:   pp512/tg128 combined, GGML_SYCL_SKIP_LMHEAD=1 + timer
# Then prints the lm_head fused-group share from the two tg numbers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

# oneAPI runtime env (Level Zero loader etc.) — required or llama-bench aborts
# "No device of requested type available" (known champion-env fragility).
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

PBIN="$ROOT/src-lmhead-build/bin"
export LX_BIN="$PBIN"
export LX_LLAMA_BENCH="$PBIN/llama-bench"
export LX_LLAMA_SERVER="$PBIN/llama-server"
export LX_LLAMA_CLI="$PBIN/llama-cli"
export LD_LIBRARY_PATH="$PBIN:$LD_LIBRARY_PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/lmhead-probe-$STAMP"
mkdir -p "$OUT"
LOG="$OUT/probe.log"

COMMON=(
  -m "$LX_MODEL" -ngl "$NGL" --n-cpu-moe "$LX_CPU_MOE_LAYERS"
  --split-mode "$LX_SPLIT_MODE" --main-gpu "$LX_MAIN_GPU" --tensor-split "$LX_TENSOR_SPLIT"
  --device "$LX_DEVICE" -t "$THREADS" --cpu-mask "$LX_CPU_MASK" --cpu-strict "$LX_CPU_STRICT"
  -b "$BBATCH" -ub "$UBATCH" -ctk "$CTK" -ctv "$CTV"
  --no-kv-offload "$LX_NO_KV_OFFLOAD" --no-op-offload "$LX_NO_OP_OFFLOAD" --no-host "$LX_NO_HOST"
  -r "$LX_REPS" -d "$LX_DEPTH" --prio "$LX_PRIO" --load-mode "$LX_LOAD_MODE" --poll "$LX_POLL"
  -o json
)

echo "-- 1) golden gate on probe binary (default env, inert) --" | tee -a "$LOG"
bash "$ROOT/scripts/golden-smoke.sh" 2>&1 | tee -a "$LOG"

lx_gpu_lock_enter "lmhead-probe" || exit $?
trap 'lx_gpu_lock_leave' EXIT

echo "-- 2) timer run: pp512/tg128 --" | tee -a "$LOG"
export GGML_SYCL_LMHEAD_TIMER=1
unset GGML_SYCL_SKIP_LMHEAD || true
"$LX_LLAMA_BENCH" "${COMMON[@]}" -pg "${LX_PP},${LX_TG}" 2>>"$LOG" > "$OUT/run1.json" || { echo "run1 failed" >&2; exit 1; }

echo "-- 3) skip run: lm_head fused group removed --" | tee -a "$LOG"
export GGML_SYCL_SKIP_LMHEAD=1
"$LX_LLAMA_BENCH" "${COMMON[@]}" -pg "${LX_PP},${LX_TG}" 2>>"$LOG" > "$OUT/run2.json" || { echo "run2 failed" >&2; exit 1; }

python3 - "$OUT" "$LOG" <<'PYEOF'
import json, sys, os
out, log = sys.argv[1], sys.argv[2]
def tg_tps(fn):
    d = json.load(open(fn))
    for r in d.get('results', []):
        if r.get('test', '') == 'tg' or r.get('n', 0) > 0 and r.get('n_prompt', 0) == 0:
            return r['avg_ts']
    # fallback: combined row
    for r in d.get('results', []):
        if r.get('test', '') == 'tg':
            return r['avg_ts']
    return None
r1, r2 = tg_tps(os.path.join(out, 'run1.json')), tg_tps(os.path.join(out, 'run2.json'))
if r1 and r2:
    cycle1 = 1e6 / r1; cycle2 = 1e6 / r2
    share = (cycle1 - cycle2) / cycle1 * 100.0
    summary = {
        'run1_tg_tps': r1, 'run2_tg_tps': r2,
        'run1_token_us': cycle1, 'run2_token_us': cycle2,
        'lmhead_fused_share_pct': share,
    }
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('SUMMARY: run1 tg=%.2f t/s (%.1f us/tok)  run2 tg=%.2f t/s (%.1f us/tok)' % (r1, cycle1, r2, cycle2))
    print('LMHEAD_FUSED_SHARE = %.2f %% of decode budget' % share)
else:
    print('could not parse tg rows (r1=%r r2=%r)' % (r1, r2))
PYEOF
echo "PROBE_DONE $OUT" | tee -a "$LOG"
