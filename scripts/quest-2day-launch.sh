#!/usr/bin/env bash
# Launch 2-day Mount Doom quest detached.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEST_DIR="${QUEST_DIR:-$ROOT/results/quest-2day-20260731}"
mkdir -p "$QUEST_DIR/logs"
LOG="$QUEST_DIR/logs/launcher-$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f "$QUEST_DIR/quest.pid" ]]; then
  old=$(cat "$QUEST_DIR/quest.pid" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping old quest-2day pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
fi

# shellcheck disable=SC1091
source "$ROOT/env.sh"
export QUEST_DIR
export QUEST_DEADLINE_UTC="${QUEST_DEADLINE_UTC:-2026-08-02T14:00:00Z}"
export QUEST_GOAL_SCORE="${QUEST_GOAL_SCORE:-1.250}"
export QUEST_STRETCH_SCORE="${QUEST_STRETCH_SCORE:-1.300}"
export QUEST_SLEEP_S="${QUEST_SLEEP_S:-1200}"
export QUEST_GOLDEN_EVERY="${QUEST_GOLDEN_EVERY:-4}"

nohup bash "$ROOT/scripts/quest-2day.sh" >>"$LOG" 2>&1 &
echo $! | tee "$QUEST_DIR/launcher.pid"
echo "quest-2day launched pid=$(cat "$QUEST_DIR/launcher.pid") log=$LOG"
echo "goal: score ≥ $QUEST_GOAL_SCORE by $QUEST_DEADLINE_UTC"
echo "tail: tail -f $QUEST_DIR/quest.log"
echo "board: cat $QUEST_DIR/BOARD.json | head"
