#!/usr/bin/env bash
# no-pinned-cycle.sh — first bench of GGML_SYCL_NO_PINNED=1 (never-benched knob).
# 3-arm same-window sandwich: ctrl-a (champion env) | cand (+NO_PINNED=1) | ctrl-b.
# Official board geometry (LATEST_SCORE.json candidate_meta).
# NO_PINNED disables zeMemAllocHost pinned host buffers — buffer-placement flag,
# output-invisible by construction (no golden gate needed, env-only).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/results/src-repro-20260806T035656Z/bin/llama-bench"
MODEL="/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf"
STAMP="no-pinned-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/$STAMP"; mkdir -p "$OUT"

"$ROOT/scripts/with-gpu-lock" --wait -- bash -c "
set -uo pipefail
BIN='$BIN'; MODEL='$MODEL'; OUT='$OUT'
FLAGS=( -m \"\$MODEL\" -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 -r 5 -o json )
BASE_ENV=( ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1 )
run(){ local label=\"\$1\"; shift; env \"\${BASE_ENV[@]}\" \"\$@\" \"\$BIN\" \"\${FLAGS[@]}\" > \"\$OUT/\$label.log\" 2>\"\$OUT/\$label.stderr\"; echo \"arm \$label rc=\$?\" >> \"\$OUT/cycle.log\"; }
run ctrl-a
run cand env GGML_SYCL_NO_PINNED=1
run ctrl-b
"
echo "cycle done rc=$?" >> "$OUT/cycle.log"

python3 - "$OUT" <<'PY'
import json,sys
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
    except Exception as e:
        print("PARSE-FAIL",p,e); return None
    if not objs: return None
    # array order is [pp, tg] on this build (verified: first avg_ts ~1165=pp, second ~138=tg)
    return {"pp": objs[0].get("avg_ts"), "tg": objs[1].get("avg_ts") if len(objs)>1 else None}
for lbl in ("ctrl-a","cand","ctrl-b"):
    r=parse(f"{out}/{lbl}.log")
    print(lbl, "OK" if r else "FAIL", r, f"{out}/{lbl}.log")
PY
