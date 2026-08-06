#!/usr/bin/env bash
# moe-dual-reenable.sh — A/B the two MoE dual-path knobs that env.sh kills as
# "quality-suspect" but that the fusion-policy audit shows were NEVER benched
# (artifact counts 0 for both DISABLE_*=0 values; prior ppl-enable data showed
# only_MOE_DUAL_MULTITOKEN PPL ratio 1.0004 — essentially neutral).
#
# Arms (one window, single lock, ctrl sandwich):
#   ctrl      = champion env (both DISABLE_* =1, as env.sh ships)
#   dualmt    = GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=0  (prefill ffn_out multi-token)
#   dualdown  = GGML_SYCL_DISABLE_MOE_DUAL_DOWN=0         (dual down-proj fusion)
#   ctrl2     = champion env again
# Each arm is a fresh llama-bench process, so a crash stays per-process and the
# next arm still runs (self-healing wedge, cf. finding 11/12).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 2

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/moe-dual-$STAMP"
mkdir -p "$OUT"

# --- champion binary (from LATEST_SCORE.json candidate_meta.binary) ----------
BENCH="${LX_BENCH:-$HERE/results/src-repro-20260806T035656Z/bin/llama-bench}"
if [[ ! -x "$BENCH" ]]; then
  BENCH="$(command -v llama-bench 2>/dev/null || true)"
fi
if [[ -z "$BENCH" || ! -x "$BENCH" ]]; then
  echo "FATAL: no llama-bench binary (tried LX_BENCH, champion path, PATH)" >&2
  exit 3
fi

# --- champion environment (LD_LIBRARY_PATH, oneapi, graph/dnn knobs) ---------
source "$HERE/env.sh" || { echo "FATAL: env.sh source failed" >&2; exit 3; }

MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
GEOM=( -m "$MODEL" -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 \
       -ctk f16 -ctv f16 -p 512 -n 128 -r 5 -o json )
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0

# --- golden smoke gate (best-effort; runtime-env arm, prior PPL ratio 1.0004) --
if [[ -x "$HERE/scripts/golden-smoke.sh" ]]; then
  echo "GOLDEN: $STAMP" | tee "$OUT/golden.log"
  if ! "$HERE/scripts/golden-smoke.sh" >> "$OUT/golden.log" 2>&1; then
    echo "FATAL: golden smoke failed" >&2
    exit 4
  fi
  echo "GOLDEN OK" | tee -a "$OUT/golden.log"
else
  echo "GOLDEN: scripts/golden-smoke.sh absent — skipping (env-only arm)" | tee "$OUT/golden.log"
fi

run_arm() { # name env_var env_val
  local name="$1" var="$2" val="$3"
  if [[ -n "$var" ]]; then
    export "$var=$val"
  else
    export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
    export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
  fi
  env | grep -E 'MOE_DUAL|MOE_PIPELINE|MOE_DOWN_GROUPED' | sort > "$OUT/$name.env"
  echo "=== ARM $name at $(date -u +%H:%M:%SZ)" | tee -a "$OUT/run.log"
  "$BENCH" "${GEOM[@]}" > "$OUT/$name.json" 2> "$OUT/$name.stderr"
  local rc=$?
  echo "rc=$rc" | tee -a "$OUT/run.log"
  if [[ $rc -ne 0 ]]; then
    echo "ARM $name rc=$rc (wedge is per-process; continuing)" | tee -a "$OUT/run.log"
  fi
  return 0
}

run_arm ctrl     ""                       ""
run_arm dualmt   GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN 0
run_arm dualdown GGML_SYCL_DISABLE_MOE_DUAL_DOWN 0
run_arm ctrl2    ""                       ""

# --- ledger ------------------------------------------------------------------
echo "STAMP=$STAMP BENCH=$BENCH" > "$OUT/LEDGER.txt"
for a in ctrl dualmt dualdown ctrl2; do
  if [[ -s "$OUT/$a.json" ]]; then
    echo "$a: $(grep -o '"avg_ts":[0-9.]*' "$OUT/$a.json" | tr '\n' ' ')" >> "$OUT/LEDGER.txt"
  else
    echo "$a: (no json)" >> "$OUT/LEDGER.txt"
  fi
done
echo "done: $OUT" | tee -a "$OUT/run.log"
