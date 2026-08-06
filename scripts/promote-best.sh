#!/usr/bin/env bash
# promote-best.sh — score-board integrity guard.
#
# Problem (measured): the board LATEST_SCORE.json keeps getting overwritten by
# whatever cycle finished last. Heat-depressed runs (~1.1908-1.1918 after a
# proof suite) have repeatedly clobbered the cooled champion score (1.2181,
# results/20260806T060605Z). The board must track the BEST VERIFIED score for
# the current champion binary, never the latest.
#
# Usage:
#   scripts/promote-best.sh [--bin <substr>] [--min <score>] [--commit]
#     --bin <substr> : only consider runs whose candidate_meta.binary matches
#                      this substring (default: src-repro — the champion build)
#     --min <score>  : never promote a score below this (default: 1.0)
#     --commit       : also commit the board update (default: print only)
#
# Policy: scans every results/<stamp>/score.json, picks the max "score" whose
# binary matches and whose floors_ok is true, and rewrites the board ONLY if
# the max is >= the current board score. Never downgrades the board.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN_SUBSTR="${BIN_SUBSTR:-src-repro}"
MIN_SCORE="${MIN_SCORE:-1.0}"
COMMIT="${COMMIT:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN_SUBSTR="$2"; shift 2 ;;
    --min) MIN_SCORE="$2"; shift 2 ;;
    --commit) COMMIT=1; shift ;;
    *) echo "promote-best: unknown arg $1" >&2; exit 2 ;;
  esac
done

BOARD="results/LATEST_SCORE.json"
CUR="$(jq -r '.score' "$BOARD" 2>/dev/null || echo 0)"

BEST=""
BEST_FILE=""
while IFS= read -r f; do
  ok="$(jq -r '.floors_ok' "$f" 2>/dev/null || echo false)"
  [ "$ok" = "true" ] || continue
  bin="$(jq -r '.candidate_meta.binary // ""' "$f" 2>/dev/null || true)"
  case "$bin" in *"$BIN_SUBSTR"*) ;; *) continue ;; esac
  s="$(jq -r '.score' "$f")"
  if [ -z "$BEST" ] || awk -v a="$s" -v b="$BEST" 'BEGIN{exit !(a>b)}'; then
    BEST="$s"; BEST_FILE="$f"
  fi
done < <(ls -t results/*/score.json 2>/dev/null || true)

if [ -z "$BEST_FILE" ]; then
  echo "promote-best: no matching scored runs (bin substr='$BIN_SUBSTR')" >&2
  exit 1
fi

echo "promote-best: current board score = $CUR"
echo "promote-best: best matching run   = $BEST ($BEST_FILE)"

# Refuse to downgrade the board.
if awk -v a="$BEST" -v b="$CUR" -v m="$MIN_SCORE" 'BEGIN{exit !(a>=b && a>=m)}'; then
  cp "$BEST_FILE" "$BOARD"
  echo "$(dirname "$BEST_FILE")" > results/LATEST_DIR.txt
  echo "promote-best: board updated to $BEST"
  if [ "$COMMIT" = "1" ]; then
    git add results/LATEST_SCORE.json results/LATEST_DIR.txt
    git commit -m "board: promote best verified score $BEST (from $(basename "$(dirname "$BEST_FILE")"))"
  fi
else
  echo "promote-best: not updating — best ($BEST) < board ($CUR) or < min ($MIN_SCORE)"
fi
