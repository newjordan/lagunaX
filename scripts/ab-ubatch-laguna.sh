#!/usr/bin/env bash
# A/B the prefill micro-batch (-ub/-b) for the Laguna server at FULL 131072 context.
#
# Measures the deployed path (llama-server /completion), not llama-bench, because the
# question is specifically "is -ub 4096 safe + faster at -c 131072 on 30.3 GiB".
# Runs A,B,A,B interleaved: GT clocks can't be pinned here (no passwordless sudo), so
# interleaving is what controls for thermal/clock drift.
#
# Honors the B70 one-GPU-owner-at-a-time rule: the server is stopped between configs.

set -euo pipefail

REPO=/home/frosty40/turbo/worktrees/treebeard-pr-private-latest
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
PORT=8092
OUT=${OUT:-/home/frosty40/turbo/lx/results/ab-ubatch-$(date -u +%Y%m%dT%H%M%SZ)}
DEPTHS=${DEPTHS:-8192 32768 65536}
ROUNDS=${ROUNDS:-2}

ONEAPI=/opt/intel/oneapi
export LD_LIBRARY_PATH="$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib/"

mkdir -p "$OUT"
RESULTS="$OUT/samples.jsonl"
: > "$RESULTS"

# Match the BINARY + port, never the bare port string — `pkill -f "port 8092"` also matches
# the shell running this script and kills it.
PAT="bin/llama-server.*--port $PORT"
server_pid() { pgrep -f "$PAT" 2>/dev/null | head -1; }

stop_server() {
  pkill -f "$PAT" 2>/dev/null || true
  for _ in $(seq 1 40); do [ -z "$(server_pid)" ] && return 0; sleep 1; done
  pkill -9 -f "$PAT" 2>/dev/null || true; sleep 2
}

# start_server <ub> <b> <logfile>; returns 1 if it failed to come up (e.g. OOM)
start_server() {
  local ub=$1 b=$2 log=$3
  ( cd "$REPO" && nohup setsid ./build-positive-package/bin/llama-server \
      -m "$MODEL" --alias laguna-xs-2.1-q4-treebeard -a laguna-xs-2.1-q4-treebeard \
      -ngl 99 -fa on -ctk f16 -ctv f16 -c 131072 -np 1 -b "$b" -ub "$ub" -t 16 \
      --host 0.0.0.0 --port "$PORT" --jinja \
      --chat-template-file models/templates/poolside-Laguna-XS-2.1.jinja \
      --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0.0 -n -1 --metrics \
      >"$log" 2>&1 </dev/null & disown )
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    [ -z "$(server_pid)" ] && { echo "  !! server died during load"; return 1; }
    sleep 1
  done
  echo "  !! server never became healthy"; return 1
}

vram_gib() {
  local pid; pid=$(server_pid)
  [ -n "$pid" ] && cat /proc/"$pid"/fdinfo/* 2>/dev/null \
    | awk '/drm-resident-vram0/{s+=$2} END {printf "%.2f", s/1048576}' || echo "0"
}

# measure <label> <ub> <depth> <round>
measure() {
  local label=$1 ub=$2 depth=$3 round=$4
  # Body goes to a FILE, not an argv string: a 32K-token prompt exceeds MAX_ARG_STRLEN
  # (128 KiB per argument) and curl dies with "Argument list too long".
  local bodyf="$OUT/.body.json"
  python3 -c "
import json,sys
depth=int(sys.argv[1]); salt=sys.argv[2]
word='telemetry accelerator kernel throughput '   # ~5 tok
n=int(depth/5)+1
json.dump({'prompt': salt+' '+(word*n), 'n_predict':8, 'cache_prompt':False, 'temperature':0},
          open(sys.argv[3],'w'))
" "$depth" "r${round}-${label}-${depth}" "$bodyf"
  local resp; resp=$(curl -s --max-time 900 "http://127.0.0.1:$PORT/completion" \
      -H 'Content-Type: application/json' --data-binary @"$bodyf")
  python3 -c "
import json,sys
r=json.loads(sys.stdin.read()); t=r.get('timings',{})
pn,pms=t.get('prompt_n',0),t.get('prompt_ms',0)
rec={'label':sys.argv[1],'ub':int(sys.argv[2]),'depth':int(sys.argv[3]),'round':int(sys.argv[4]),
     'prompt_n':pn,'prompt_ms':pms,'pp_tps':(pn/(pms/1000) if pms else 0),
     'tg_tps':t.get('predicted_per_second',0),'vram_gib':float(sys.argv[5])}
print(json.dumps(rec))
sys.stderr.write('  %-6s ub=%-5s depth=%-6s pp=%7.1f t/s  tg=%5.1f t/s  vram=%sGiB\n'%(
    rec['label'],rec['ub'],rec['prompt_n'],rec['pp_tps'],rec['tg_tps'],rec['vram_gib']))
" "$label" "$ub" "$depth" "$round" "$(vram_gib)" <<<"$resp" >> "$RESULTS"
}

# Never leave a server holding the GPU if we die mid-run — the next run's lock would refuse.
trap 'echo "[cleanup] stopping server"; stop_server' EXIT INT TERM

echo "=== A/B -ub at -c 131072 -np 1 | out=$OUT ==="
stop_server
for round in $(seq 1 "$ROUNDS"); do
  for cfg in "A 2048 4096" "B 4096 8192"; do
    set -- $cfg; label=$1 ub=$2 b=$3
    echo "-- round $round | $label: -ub $ub -b $b --"
    if ! start_server "$ub" "$b" "$OUT/server-$label-r$round.log"; then
      echo "{\"label\":\"$label\",\"ub\":$ub,\"round\":$round,\"failed\":true}" >> "$RESULTS"
      grep -iE "error|oom|alloc|failed" "$OUT/server-$label-r$round.log" | tail -5 || true
      stop_server; continue
    fi
    echo "  loaded, vram=$(vram_gib) GiB"
    for d in $DEPTHS; do measure "$label" "$ub" "$d" "$round"; done
    stop_server
  done
done

echo; echo "=== SUMMARY ==="
python3 - "$RESULTS" <<'EOF'
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1])]
rows=[r for r in rows if not r.get('failed')]
depths=sorted({r['depth'] for r in rows})
print(f"{'depth(req)':>11} {'tokens':>7} {'A ub2048':>10} {'B ub4096':>10} {'delta':>9}")
for d in depths:
    a=[r['pp_tps'] for r in rows if r['depth']==d and r['label']=='A']
    b=[r['pp_tps'] for r in rows if r['depth']==d and r['label']=='B']
    n=[r['prompt_n'] for r in rows if r['depth']==d]
    if not a or not b: continue
    ma,mb=st.mean(a),st.mean(b)
    print(f"{d:>11} {n[0]:>7} {ma:>10.1f} {mb:>10.1f} {(mb-ma)/ma*100:>+8.1f}%")
for lab,ub in (('A',2048),('B',4096)):
    v=[r['vram_gib'] for r in rows if r['label']==lab]
    if v: print(f"  {lab} ub={ub}: peak VRAM {max(v):.2f} GiB / 30.3 GiB")
EOF
echo "raw: $RESULTS"
