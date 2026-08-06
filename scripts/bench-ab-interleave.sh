#!/usr/bin/env bash
# A/B interleaver: champion binary vs env-only candidate, alternating inside ONE
# lock window, so deltas are judged against within-window drift (observed between-
# run drift ~0.7% tg is ~9x the within-run SD). With ROUNDS of A,B,A,B the mean
# delta separates a +3.5% tg / +5% pp win at ~5x the drift bound.
#
# Usage:
#   bash scripts/bench-ab-interleave.sh <champ-bin> <CAND_ENV_SPEC> [rounds=2] [note]
#   CAND_ENV_SPEC = e.g. GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1 (exported only for B runs)
#   LX_GPU_LOCK_WAIT=<secs> to queue behind a running holder (default 0 = refuse).
# Output: results/ab-<stamp>/{table.tsv,receipt.json}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
source "$ROOT/env.sh"
# shellcheck source=scripts/lib-gpu-lock.sh
source "$ROOT/scripts/lib-gpu-lock.sh"

CHAMP_BIN="${1:?champion llama-bench binary}"
CAND_SPEC="${2:?candidate env spec, e.g. GGML_SYCL_X=1}"
ROUNDS="${3:-2}"
NOTE="${4:-}"

CAND_NAME="${CAND_SPEC%%=*}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$LX_RESULTS/ab-${STAMP}"
mkdir -p "$OUT"
TABLE="$OUT/table.tsv"

echo "== [ab] $STAMP champion=$CHAMP_BIN candidate=$CAND_SPEC rounds=$ROUNDS"
echo "  out: $OUT"

lx_gpu_lock_enter "ab-${STAMP}" || {
  echo "[ab] lock refused/queued; see .b70-gpu.lock meta" >&2
  exit 75
}
trap lx_gpu_lock_leave EXIT

# Exact official flag set (bench-serial.sh COMMON), combined pp+tg in one process.
FLAGS=(
  -m "$LX_MODEL" -ngl "$NGL"
  --n-cpu-moe "$LX_CPU_MOE_LAYERS" --split-mode "$LX_SPLIT_MODE"
  --main-gpu "$LX_MAIN_GPU" --tensor-split "$LX_TENSOR_SPLIT"
  --device "$LX_DEVICE" -t "$THREADS" --cpu-mask "$LX_CPU_MASK"
  --cpu-strict "$LX_CPU_STRICT" -b "$BBATCH" -ub "$UBATCH"
  -ctk "$CTK" -ctv "$CTV"
  --no-kv-offload "$LX_NO_KV_OFFLOAD" --no-op-offload "$LX_NO_OP_OFFLOAD"
  --no-host "$LX_NO_HOST"
  -r "$LX_REPS" -d "$LX_DEPTH" --prio "$LX_PRIO"
  --load-mode "$LX_LOAD_MODE" --poll "$LX_POLL" --delay "$LX_DELAY"
)
case "${FA:-}" in
  on|off|auto) FLAGS+=(-fa "$FA") ;;
esac

run_once() {
  local tag="$1" seq="$2" env_line="(champion env)"
  if [[ "$tag" == "B" ]]; then
    export "$CAND_SPEC"
    env_line="$CAND_SPEC"
  else
    unset "$CAND_NAME" 2>/dev/null || true
  fi
  echo "  [ab] run $seq $tag $env_line $(date -u +%H:%M:%S)"
  local json
  json="$("$CHAMP_BIN" "${FLAGS[@]}" -pg "$LX_PP,$LX_TG" -o json 2>>"$OUT/llama-bench.log")"
  local pp tg pp_sd tg_sd
  pp="$(printf '%s' "$json" | python3 -c 'import json,sys
r=json.load(sys.stdin)
print([x["avg_ts"] for x in r if x["n_prompt"]==512 and x["n_gen"]==0][0])')"
  tg="$(printf '%s' "$json" | python3 -c 'import json,sys
r=json.load(sys.stdin)
print([x["avg_ts"] for x in r if x["n_prompt"]==0 and x["n_gen"]==128][0])')"
  pp_sd="$(printf '%s' "$json" | python3 -c 'import json,sys
r=json.load(sys.stdin)
print([x["stddev_ts"] for x in r if x["n_prompt"]==512 and x["n_gen"]==0][0])')"
  tg_sd="$(printf '%s' "$json" | python3 -c 'import json,sys
r=json.load(sys.stdin)
print([x["stddev_ts"] for x in r if x["n_prompt"]==0 and x["n_gen"]==128][0])')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$tag" "$pp" "$tg" "$pp_sd" "$tg_sd" >>"$TABLE"
}

# A,B,A,B,... — alternating kills slow thermal drift by construction.
for ((i = 1; i <= ROUNDS; i++)); do
  run_once A "$i"
  run_once B "$i"
done

python3 - "$TABLE" "$CAND_SPEC" "$CHAMP_BIN" "$ROUNDS" "$STAMP" "$NOTE" <<'PY' >"$OUT/receipt.json"
import json, sys, statistics
table, cand, champ, rounds, stamp, note = sys.argv[1:7]
rows = []
with open(table) as f:
    for ln in f:
        seq, tag, pp, tg, ppsd, tgsd = ln.rstrip("\n").split("\t")
        rows.append(dict(seq=int(seq), tag=tag, pp=float(pp), tg=float(tg),
                         pp_sd=float(ppsd), tg_sd=float(tgsd)))
A = [r for r in rows if r["tag"] == "A"]
B = [r for r in rows if r["tag"] == "B"]
def mean(xs): return statistics.mean(xs)
def sdev(xs): return statistics.stdev(xs) if len(xs) > 1 else 0.0
ppA, ppB = mean([r["pp"] for r in A]), mean([r["pp"] for r in B])
tgA, tgB = mean([r["tg"] for r in A]), mean([r["tg"] for r in B])
pp_d = (ppB - ppA) / ppA * 100
tg_d = (tgB - tgA) / tgA * 100
pp_jit = sdev([r["pp"] for r in A] + [r["pp"] for r in B])
tg_jit = sdev([r["tg"] for r in A] + [r["tg"] for r in B])
verdict = {
    "tg_delta_pct": round(tg_d, 3),
    "pp_delta_pct": round(pp_d, 3),
    "tg_jitter_sd_pct": round(tg_jit / tgA * 100, 3),
    "pp_jitter_sd_pct": round(pp_jit / ppA * 100, 3),
    "tg_signal_over_drift": round(abs(tg_d) / 0.68, 2),
    "pp_signal_over_drift": round(abs(pp_d) / 1.5, 2),
    "verdict": ("WIN" if tg_d >= 1.5 and abs(tg_d) >= 3 * tg_jit / tgA * 100 else
                "REG" if tg_d <= -1.5 else "NOISE"),
}
out = dict(stamp=stamp, champion_bin=champ, candidate_env=cand, rounds=int(rounds),
           note=note, runs=rows, champion=dict(pp=round(ppA, 3), tg=round(tgA, 3)),
           candidate=dict(pp=round(ppB, 3), tg=round(tgB, 3)), **verdict)
print(json.dumps(out, indent=2))
PY
echo "== [ab] receipt: $OUT/receipt.json"
cat "$OUT/receipt.json"
