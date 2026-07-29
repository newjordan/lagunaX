#!/usr/bin/env bash
# Launch Mount Doom quest detached (survives terminal disconnect).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEST_DIR="${QUEST_DIR:-$ROOT/results/quest-mount-doom}"
mkdir -p "$QUEST_DIR/logs"
LOG="$QUEST_DIR/logs/launcher-$(date -u +%Y%m%dT%H%M%SZ).log"

# kill prior quest if any
if [[ -f "$QUEST_DIR/quest.pid" ]]; then
  old=$(cat "$QUEST_DIR/quest.pid" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping old quest pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
fi

nohup bash "$ROOT/scripts/quest-mount-doom.sh" >>"$LOG" 2>&1 &
echo $! | tee "$QUEST_DIR/launcher.pid"
echo "quest launched pid=$(cat $QUEST_DIR/launcher.pid) log=$LOG"
echo "tail: tail -f $QUEST_DIR/quest.log"
