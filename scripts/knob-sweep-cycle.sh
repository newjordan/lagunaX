#!/usr/bin/env bash
# knob-sweep-cycle.sh — FIRST complete env-knob coverage sweep.
# Each never-benched env knob gets a same-window 3-arm sandwich:
#   ctrl-a (champion env) | cand (+knob) | ctrl-b
# Screening geometry: official flags, -r 3 (fast). Any |delta|>1.0% vs ctrl
# mean is flagged CANDIDATE and promoted to the official r=5 pipeline later.
# Never-benched knobs (from getenv inventory, 40 total, 16 unbenched):
#   FATTN_FORCE_TILE / ADD_ADD / MUL_MAT_ADD / RMS_NORM / DENSE_DUAL_SWIGLU /
#   MOE_DUAL_SWIGLU / SOFTPLUS_MUL / ROPE_SET_ROWS / MMID_FUSED_BATCH /
#   ROUTER_SIGMOID_ADD   (NO_PINNED benched separately: null; SKIP_LMHEAD /
#   DISABLE_TOPK_MOE / GRAPH_CHECKSUM / LMHEAD_KPATH are kill/debug knobs, out)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/results/src-repro-20260806T035656Z/bin/llama-bench"
MODEL="/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf"
STAMP="knob-sweep-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/$STAMP"; mkdir -p "$OUT"
KNOBS=(
  "GGML_SYCL_FATTN_FORCE_TILE=1"
  "GGML_SYCL_DISABLE_ADD_ADD_FUSE=1"
  "GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1"
  "GGML_SYCL_DISABLE_RMS_NORM_FUSE=1"
  "GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1"
  "GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1"
  "GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE=1"
  "GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE=1"
  "GGML_SYCL_DISABLE_MMID_FUSED_BATCH=1"
  "GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD=1"
)
export BIN MODEL OUT
KNOBS_LIST="$(printf '%s\n' "${KNOBS[@]}")"
export KNOBS_LIST
"$ROOT/scripts/with-gpu-lock" --wait -- bash -c '
  set -uo pipefail
  FLAGS=( -m "$MODEL" -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto \
          -b 2048 -ub 2048 -ctk f16 -ctv f16 -r 3 -o json )
  BASE_ENV=( ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 \
             GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1 )
  run(){ local label="$1"; shift; env "${BASE_ENV[@]}" "$@" "$BIN" "${FLAGS[@]}" \
        > "$OUT/$label.log" 2>"$OUT/$label.stderr" || echo "arm $label FAILED rc=$?"; }
  while IFS= read -r knob; do
    [ -z "$knob" ] && continue
    echo "=== knob: $knob ===" >> "$OUT/summary.txt"
    run "${knob%%=*}-a"
    run "${knob%%=*}-cand" env "$knob"
    run "${knob%%=*}-b"
  done <<< "$KNOBS_LIST"
'
echo "sweep rc=$?" >> "$OUT/summary.txt"
python3 - "$OUT" <<'PY'
import json,sys,os
out=sys.argv[1]
def parse(p):
    txt=open(p,errors="replace").read()
    txt="\n".join(l for l in txt.splitlines() if not l.strip().startswith("[lx-"))
    dec=json.JSONDecoder(); idx=0; objs=[]
    try:
        while idx < len(txt):
            while idx < len(txt) and txt[idx] not in "{[": idx+=1
            if idx>=len(txt): break
            o,end=dec.raw_decode(txt,idx)
            if isinstance(o,list): objs.extend(o)
            else: objs.append(o)
            idx=end
    except Exception: return None
    if len(objs)<2: return None
    return {"pp": objs[0].get("avg_ts"), "tg": objs[1].get("avg_ts")}
names=sorted(set(f.split("-")[0] for f in os.listdir(out) if f.endswith("-cand.log")))
print("knob,tg_cand,tg_ctrl_mean,tg_delta_pct,pp_cand,pp_ctrl_mean,pp_delta_pct,verdict")
for n in names:
    c=parse(f"{out}/{n}-cand.log"); a=parse(f"{out}/{n}-a.log"); b=parse(f"{out}/{n}-b.log")
    if not (c and a and b):
        print(f"{n},FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL"); continue
    tm=(a["tg"]+b["tg"])/2; pm=(a["pp"]+b["pp"])/2
    td=(c["tg"]-tm)/tm*100; pd=(c["pp"]-pm)/pm*100
    verdict="CANDIDATE" if abs(td)>1.0 or abs(pd)>1.0 else "null"
    print(f"{n},{c['tg']:.3f},{tm:.3f},{td:+.3f},{c['pp']:.3f},{pm:.3f},{pd:+.3f},{verdict}")
PY