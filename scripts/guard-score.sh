#!/usr/bin/env bash
# guard-score.sh — regression gate for the serial loop.
# Fails (exit 1) if the latest official run regressed below the tip-freeze bar
# or floored, so a loop never silently ships a worse binary. Read-only.
# Usage: ./scripts/guard-score.sh [bar]   (default bar 1.209 = tip-freeze QS bar)
set -uo pipefail
cd "$(dirname "$0")/.."
BAR="${1:-1.209}"

SCORE_JSON="results/LATEST_SCORE.json"
if [[ ! -f "$SCORE_JSON" ]]; then
  echo "GUARD FAIL: $SCORE_JSON missing" >&2
  exit 1
fi

score=$(jq -r '.score' "$SCORE_JSON")
floors=$(jq -r '.floors_ok' "$SCORE_JSON")
increase=$(jq -r '.increase_pct' "$SCORE_JSON")
decode=$(jq -r '.decode_speedup' "$SCORE_JSON")
prefill=$(jq -r '.prefill_speedup' "$SCORE_JSON")

echo "guard: score=$score (+${increase}%) decode_speedup=$decode prefill_speedup=$prefill floors_ok=$floors bar=$BAR"

if [[ "$floors" != "true" ]]; then
  echo "GUARD FAIL: floors not ok — candidate below floor" >&2
  exit 1
fi
if ! awk -v s="$score" -v b="$BAR" 'BEGIN { exit !(s >= b) }'; then
  echo "GUARD FAIL: score $score < bar $BAR (regression vs tip-freeze)" >&2
  exit 1
fi
echo "GUARD OK: score >= bar, floors pass"
