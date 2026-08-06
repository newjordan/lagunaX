#!/usr/bin/env bash
# QUEST: Mount Doom — continuous Laguna serial absolute-limit on B70.
# Runs until killed. Cycles: kernel-trace → rebench → log → sleep → repeat.
# Safe to leave for days. Does NOT re-pin baseline. Does NOT multi-slot.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

QUEST_DIR="${QUEST_DIR:-$LX_RESULTS/quest-mount-doom}"
mkdir -p "$QUEST_DIR/cycles" "$QUEST_DIR/logs"
LOG="$QUEST_DIR/quest.log"
PIDFILE="$QUEST_DIR/quest.pid"
echo $$ >"$PIDFILE"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "QUEST START pid=$$"
log "baseline=$LX_BASELINE_JSON bin=$LX_BIN model=$LX_MODEL"

CYCLE=0
while true; do
  CYCLE=$((CYCLE + 1))
  CSTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  CDIR="$QUEST_DIR/cycles/c${CYCLE}-${CSTAMP}"
  mkdir -p "$CDIR"
  log "======== CYCLE $CYCLE → $CDIR ========"

  # free GPU of stray benches (only if we can take the exclusive lock next)
  for p in $(pgrep -x llama-bench 2>/dev/null || true); do
    # don't kill ourselves if somehow named that
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 2

  # Wait up to 15m for the card (agents may still be finishing).
  export LX_GPU_LOCK_WAIT="${LX_GPU_LOCK_WAIT:-900}"
  if ! lx_gpu_lock_enter "quest-mount-doom-c${CYCLE}"; then
    log "WARN: could not acquire B70 lock — sleeping and retrying cycle"
    sleep 60
    continue
  fi
  # Release at end of cycle body (before long sleep).

  # --- serial rebench (ship flags) ---
  export LD_LIBRARY_PATH="$LX_BIN:${LD_LIBRARY_PATH:-}"
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  export ZE_AFFINITY_MASK=0
  unset GGML_SYCL_DISABLE_DNN 2>/dev/null || true
  export GGML_SYCL_DISABLE_GRAPH=1

  PP_JSON="$("$LX_LLAMA_BENCH" -m "$LX_MODEL" -ngl 99 -t 16 -ub 4096 -b 8192 \
    -ctk f16 -ctv f16 -fa on -r 5 -p 512 -n 0 -o json 2>"$CDIR/pp.err")" || {
    log "CYCLE $CYCLE pp FAIL"; echo fail >"$CDIR/status"
    lx_gpu_lock_leave; sleep 60; continue
  }
  TG_JSON="$("$LX_LLAMA_BENCH" -m "$LX_MODEL" -ngl 99 -t 16 -ub 4096 -b 8192 \
    -ctk f16 -ctv f16 -fa on -r 5 -p 0 -n 128 -o json 2>"$CDIR/tg.err")" || {
    log "CYCLE $CYCLE tg FAIL"; echo fail >"$CDIR/status"
    lx_gpu_lock_leave; sleep 60; continue
  }

  python3 - "$CDIR/metrics.json" "$PP_JSON" "$TG_JSON" "$CYCLE" <<'PY'
import json,sys,os
from pathlib import Path
out,ppb,tgb,cycle=sys.argv[1:5]
def avg(blob,kind):
    data=json.loads(blob)
    rows=data["results"] if isinstance(data,dict) and "results" in data else (data if isinstance(data,list) else [data])
    for r in rows:
        a=r.get("avg_ts")
        if a is None: continue
        ng,np=r.get("n_gen"),r.get("n_prompt")
        t=str(r.get("test") or "")
        if kind=="pp" and (t.startswith("pp") or ng in (0,"0",None)): return float(a)
        if kind=="tg" and (t.startswith("tg") or np in (0,"0",None)): return float(a)
    return float(rows[0]["avg_ts"])
pp,tg=avg(ppb,"pp"),avg(tgb,"tg")
Path(out).write_text(json.dumps({"cycle":int(cycle),"pp512":pp,"tg128":tg},indent=2)+"\n")
print(f"pp={pp:.3f} tg={tg:.3f}")
PY
  echo ok >"$CDIR/status"

  if [[ -f "$LX_BASELINE_JSON" ]]; then
    python3 "$ROOT/scripts/score.py" \
      --baseline "$LX_BASELINE_JSON" \
      --candidate "$CDIR/metrics.json" \
      -o "$CDIR/score.json" 2>>"$LOG" || true
  fi

  PP=$(python3 -c "import json;print(json.load(open('$CDIR/metrics.json'))['pp512'])")
  TG=$(python3 -c "import json;print(json.load(open('$CDIR/metrics.json'))['tg128'])")
  SCORE=$(python3 -c "import json;print(json.load(open('$CDIR/score.json')).get('score','?'))" 2>/dev/null || echo '?')
  log "CYCLE $CYCLE RESULT pp512=$PP tg128=$TG score=$SCORE"

  # update champion if better tg
  CHAMP="$QUEST_DIR/CHAMPION.json"
  python3 - "$CHAMP" "$CDIR/metrics.json" "$CDIR/score.json" "$CDIR" <<'PY'
import json,sys
from pathlib import Path
champ_p, met_p, sc_p, cdir = map(Path, sys.argv[1:5])
m=json.loads(met_p.read_text())
s=json.loads(sc_p.read_text()) if sc_p.exists() else {}
cand={"pp512":m["pp512"],"tg128":m["tg128"],"score":s.get("score"),"cycle":m.get("cycle"),"dir":str(cdir)}
if champ_p.exists():
    ch=json.loads(champ_p.read_text())
    # champion by tg128 then pp then score
    better = (cand["tg128"], cand["pp512"], cand.get("score") or 0) > (ch["tg128"], ch["pp512"], ch.get("score") or 0)
else:
    better=True
if better:
    champ_p.write_text(json.dumps(cand,indent=2)+"\n")
    print("NEW CHAMPION", cand)
else:
    print("champ holds", json.loads(champ_p.read_text()) if champ_p.exists() else {})
PY

  # every 3rd cycle: quick power profile decode
  if (( CYCLE % 3 == 0 )); then
    log "CYCLE $CYCLE power profile"
    bash ~/.claude/skills/b70-profile/profile.sh \
      --model "$LX_MODEL" --mode decode --bin "$LX_BIN" \
      --out "$CDIR/profile-decode" >>"$CDIR/profile.txt" 2>&1 || true
  fi

  # every 6th cycle: oneDNN shape snapshot (decode short)
  if (( CYCLE % 6 == 0 )); then
    log "CYCLE $CYCLE kernel-trace onednn decode"
    bash ~/.claude/skills/b70-kernel-trace/ktrace.sh \
      --mode onednn --model "$LX_MODEL" --bin "$LX_BIN" \
      --out "$CDIR/ktrace-decode" -- \
      -p 0 -n 64 -r 1 -ub 4096 -b 8192 -fa on -t 16 \
      >>"$CDIR/ktrace.txt" 2>&1 || true
  fi

  # board rollup
  python3 - "$QUEST_DIR" <<'PY'
import json
from pathlib import Path
qd=Path(sys.argv[1]) if False else None
import sys
qd=Path(sys.argv[1])
rows=[]
for d in sorted((qd/"cycles").glob("c*")):
    m=d/"metrics.json"
    if not m.exists(): continue
    mj=json.loads(m.read_text())
    sj={}
    if (d/"score.json").exists():
        sj=json.loads((d/"score.json").read_text())
    rows.append({**mj,"score":sj.get("score"),"increase_pct":sj.get("increase_pct"),"dir":d.name})
rows=sorted(rows,key=lambda r:r.get("tg128") or 0, reverse=True)
(qd/"BOARD.json").write_text(json.dumps({"n":len(rows),"top":rows[:20],"all":rows},indent=2)+"\n")
if rows:
    best=rows[0]
    print(f"board n={len(rows)} best_tg={best.get('tg128')} best_pp={best.get('pp512')}")
PY

  # Drop exclusive lock before idle sleep so other harness jobs can use the card.
  lx_gpu_lock_leave

  # pause between cycles (keep GPU from thermal soak / leave room for kernel builds)
  SLEEP_S="${QUEST_SLEEP_S:-120}"
  log "CYCLE $CYCLE sleep ${SLEEP_S}s"
  sleep "$SLEEP_S"
done
