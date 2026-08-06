#!/usr/bin/env bash
# 2-day Mount Doom push loop — quality-safe tip rebench + board.
# Does NOT re-pin baseline. Does NOT multi-slot. One GPU owner via flock.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

QUEST_DIR="${QUEST_DIR:-$LX_RESULTS/quest-2day-20260731}"
DEADLINE_UTC="${QUEST_DEADLINE_UTC:-2026-08-02T14:00:00Z}"
GOAL_SCORE="${QUEST_GOAL_SCORE:-1.250}"
STRETCH_SCORE="${QUEST_STRETCH_SCORE:-1.300}"
SLEEP_S="${QUEST_SLEEP_S:-1200}"
GOLDEN_EVERY="${QUEST_GOLDEN_EVERY:-4}"

mkdir -p "$QUEST_DIR/cycles" "$QUEST_DIR/logs" "$QUEST_DIR/goal-hits"
LOG="$QUEST_DIR/quest.log"
PIDFILE="$QUEST_DIR/quest.pid"
echo $$ >"$PIDFILE"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

deadline_epoch() {
  date -u -d "$DEADLINE_UTC" +%s 2>/dev/null || date -u -d "${DEADLINE_UTC/Z/ UTC}" +%s 2>/dev/null || echo 0
}

DEADLINE_S=$(deadline_epoch)
NOW_S=$(date -u +%s)
if [[ "$DEADLINE_S" -gt 0 && "$NOW_S" -ge "$DEADLINE_S" ]]; then
  log "DEADLINE already passed ($DEADLINE_UTC) — exit"
  exit 0
fi

# Seed champion from live formal if present and better than empty
SEED="$LX_RESULTS/LATEST_SCORE.json"
if [[ -f "$SEED" && ! -f "$QUEST_DIR/CHAMPION.json" ]]; then
  python3 - "$SEED" "$QUEST_DIR/CHAMPION.json" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())
cand={
  "pp512": s.get("prefill_tok_s"),
  "tg128": s.get("decode_tok_s"),
  "score": s.get("score"),
  "increase_pct": s.get("increase_pct"),
  "source": "seed-LATEST_SCORE",
  "binary": (s.get("candidate_meta") or {}).get("binary"),
  "note": (s.get("candidate_meta") or {}).get("note"),
}
Path(sys.argv[2]).write_text(json.dumps(cand, indent=2)+"\n")
print("seeded champion", cand)
PY
fi

# Goal snapshot
python3 - "$QUEST_DIR/GOAL.json" "$DEADLINE_UTC" "$GOAL_SCORE" "$STRETCH_SCORE" "$LX_BIN" <<'PY'
import json,sys
from pathlib import Path
p,dl,gs,ss,binp=sys.argv[1:6]
Path(p).write_text(json.dumps({
  "deadline_utc": dl,
  "goal_score": float(gs),
  "stretch_score": float(ss),
  "start_score_target_beat": 1.227,
  "binary": binp,
  "track": "serial quality-safe tip",
  "formula": "decode_speedup^0.75 * prefill_speedup^0.25",
}, indent=2)+"\n")
PY

log "QUEST-2DAY START pid=$$"
log "deadline=$DEADLINE_UTC goal_score=$GOAL_SCORE stretch=$STRETCH_SCORE sleep=${SLEEP_S}s"
log "baseline=$LX_BASELINE_JSON bin=$LX_BIN model=$LX_MODEL"
log "env: mm-add=${GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE:-?} dual_down=${GGML_SYCL_DISABLE_MOE_DUAL_DOWN:-?} dual_mt=${GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN:-?}"

CYCLE=0
while true; do
  NOW_S=$(date -u +%s)
  if [[ "$DEADLINE_S" -gt 0 && "$NOW_S" -ge "$DEADLINE_S" ]]; then
    log "DEADLINE reached ($DEADLINE_UTC) — stopping after $CYCLE cycles"
    break
  fi
  LEFT=$(( DEADLINE_S > 0 ? DEADLINE_S - NOW_S : 0 ))
  log "time_left_s=$LEFT"

  CYCLE=$((CYCLE + 1))
  CSTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  CDIR="$QUEST_DIR/cycles/c${CYCLE}-${CSTAMP}"
  mkdir -p "$CDIR"
  log "======== CYCLE $CYCLE → $CDIR ========"

  export LX_GPU_LOCK_WAIT="${LX_GPU_LOCK_WAIT:-900}"
  if ! lx_gpu_lock_enter "quest-2day-c${CYCLE}"; then
    log "WARN: lock busy — sleep 60s"
    sleep 60
    continue
  fi

  export LD_LIBRARY_PATH="$LX_BIN:${LD_LIBRARY_PATH:-}"
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
  export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
  export GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-1}"

  # Nested harness would deadlock on same flock — we already hold the card.
  export LX_GPU_LOCK_SKIP=1

  # --- golden (periodic) ---
  if (( CYCLE % GOLDEN_EVERY == 1 )); then
    log "CYCLE $CYCLE golden-smoke"
    if bash "$ROOT/scripts/golden-smoke.sh" >"$CDIR/golden.log" 2>&1; then
      echo ok >"$CDIR/golden.status"
      log "CYCLE $CYCLE GOLDEN OK"
    else
      echo fail >"$CDIR/golden.status"
      log "CYCLE $CYCLE GOLDEN FAIL — see $CDIR/golden.log (no champion update this cycle)"
      unset LX_GPU_LOCK_SKIP
      lx_gpu_lock_leave
      sleep "$SLEEP_S"
      continue
    fi
  fi

  # --- formal serial (ship flags from env.sh via bench-serial) ---
  if ! bash "$ROOT/scripts/bench-serial.sh" --note "quest-2day c${CYCLE}" >"$CDIR/bench.log" 2>&1; then
    log "CYCLE $CYCLE bench FAIL"
    echo fail >"$CDIR/status"
    unset LX_GPU_LOCK_SKIP
    lx_gpu_lock_leave
    sleep 60
    continue
  fi
  unset LX_GPU_LOCK_SKIP

  # copy latest formal into cycle dir
  if [[ -f "$LX_RESULTS/LATEST_DIR.txt" ]]; then
    LDIR=$(cat "$LX_RESULTS/LATEST_DIR.txt")
    cp -a "$LDIR/metrics.json" "$CDIR/metrics.json" 2>/dev/null || true
    cp -a "$LDIR/score.json" "$CDIR/score.json" 2>/dev/null || true
    echo "$LDIR" >"$CDIR/LATEST_DIR.txt"
  fi
  echo ok >"$CDIR/status"

  PP=$(python3 -c "import json;print(json.load(open('$CDIR/metrics.json'))['pp512'])" 2>/dev/null || echo '?')
  TG=$(python3 -c "import json;print(json.load(open('$CDIR/metrics.json'))['tg128'])" 2>/dev/null || echo '?')
  SCORE=$(python3 -c "import json;print(json.load(open('$CDIR/score.json')).get('score','?'))" 2>/dev/null || echo '?')
  INC=$(python3 -c "import json;print(json.load(open('$CDIR/score.json')).get('increase_pct','?'))" 2>/dev/null || echo '?')
  log "CYCLE $CYCLE RESULT pp512=$PP tg128=$TG score=$SCORE increase=$INC%"

  # --- champion by score then tg then pp ---
  python3 - "$QUEST_DIR/CHAMPION.json" "$CDIR/metrics.json" "$CDIR/score.json" "$CDIR" "$GOAL_SCORE" "$STRETCH_SCORE" "$QUEST_DIR" <<'PY'
import json, sys
from pathlib import Path
champ_p = Path(sys.argv[1])
met_p = Path(sys.argv[2])
sc_p = Path(sys.argv[3])
cdir = Path(sys.argv[4])
goal = float(sys.argv[5])
stretch = float(sys.argv[6])
qdir = Path(sys.argv[7])
m = json.loads(met_p.read_text()) if met_p.exists() else {}
s = json.loads(sc_p.read_text()) if sc_p.exists() else {}
cand = {
    "pp512": m.get("pp512") or m.get("prefill_tok_s"),
    "tg128": m.get("tg128") or m.get("decode_tok_s"),
    "score": s.get("score"),
    "increase_pct": s.get("increase_pct"),
    "floors_ok": s.get("floors_ok"),
    "cycle_dir": str(cdir),
    "binary": m.get("binary") or (s.get("candidate_meta") or {}).get("binary"),
    "note": m.get("note"),
}
if cand["score"] is None or cand["tg128"] is None:
    print("skip champ: incomplete metrics")
    raise SystemExit(0)
if champ_p.exists():
    ch = json.loads(champ_p.read_text())
    better = (
        (cand["score"] or 0),
        (cand["tg128"] or 0),
        (cand["pp512"] or 0),
    ) > (
        (ch.get("score") or 0),
        (ch.get("tg128") or 0),
        (ch.get("pp512") or 0),
    )
else:
    better = True
if better and s.get("floors_ok", True):
    champ_p.write_text(json.dumps(cand, indent=2) + "\n")
    print("NEW CHAMPION", cand)
else:
    print("champ holds", json.loads(champ_p.read_text()) if champ_p.exists() else {})

sc = float(cand["score"] or 0)
if sc >= goal:
    hit = qdir / "goal-hits" / f"goal-{cdir.name}.json"
    hit.parent.mkdir(parents=True, exist_ok=True)
    hit.write_text(json.dumps({"level": "GOAL" if sc < stretch else "STRETCH", **cand}, indent=2) + "\n")
    print("*** GOAL HIT ***" if sc < stretch else "*** STRETCH HIT ***", sc)
PY

  # power / ktrace occasionally (still under lock)
  if (( CYCLE % 3 == 0 )); then
    log "CYCLE $CYCLE power profile decode"
    bash ~/.claude/skills/b70-profile/profile.sh \
      --model "$LX_MODEL" --mode decode --bin "$LX_BIN" \
      --out "$CDIR/profile-decode" >>"$CDIR/profile.txt" 2>&1 || true
  fi
  if (( CYCLE % 6 == 0 )); then
    log "CYCLE $CYCLE ktrace onednn decode"
    bash ~/.claude/skills/b70-kernel-trace/ktrace.sh \
      --mode onednn --model "$LX_MODEL" --bin "$LX_BIN" \
      --out "$CDIR/ktrace-decode" -- \
      -p 0 -n 64 -r 1 -ub "${UBATCH}" -b "${BBATCH}" -fa on -t "${THREADS}" \
      >>"$CDIR/ktrace.txt" 2>&1 || true
  fi

  # board rollup
  python3 - "$QUEST_DIR" "$GOAL_SCORE" "$DEADLINE_UTC" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
qd = Path(sys.argv[1])
goal = float(sys.argv[2])
deadline = sys.argv[3]
rows = []
for d in sorted((qd / "cycles").glob("c*")):
    m, s = d / "metrics.json", d / "score.json"
    if not m.exists():
        continue
    mj = json.loads(m.read_text())
    sj = json.loads(s.read_text()) if s.exists() else {}
    rows.append({
        "dir": d.name,
        "pp512": mj.get("pp512") or mj.get("prefill_tok_s"),
        "tg128": mj.get("tg128") or mj.get("decode_tok_s"),
        "score": sj.get("score"),
        "increase_pct": sj.get("increase_pct"),
        "floors_ok": sj.get("floors_ok"),
        "golden": (d / "golden.status").read_text().strip() if (d / "golden.status").exists() else None,
    })
rows_s = sorted(rows, key=lambda r: (r.get("score") or 0), reverse=True)
champ = {}
if (qd / "CHAMPION.json").exists():
    champ = json.loads((qd / "CHAMPION.json").read_text())
board = {
    "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "deadline_utc": deadline,
    "goal_score": goal,
    "n_cycles": len(rows),
    "champion": champ,
    "goal_gap": (goal - (champ.get("score") or 0)) if champ.get("score") is not None else None,
    "top": rows_s[:25],
}
(qd / "BOARD.json").write_text(json.dumps(board, indent=2) + "\n")
print(f"board n={len(rows)} champ_score={champ.get('score')} gap_to_goal={board['goal_gap']}")
PY

  lx_gpu_lock_leave
  log "CYCLE $CYCLE sleep ${SLEEP_S}s"
  sleep "$SLEEP_S"
done

log "QUEST-2DAY DONE cycles=$CYCLE"
rm -f "$PIDFILE"
