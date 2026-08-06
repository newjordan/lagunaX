#!/usr/bin/env bash
# knob-ab-ledger.sh — collapse every interleaved A/B receipt (results/ab-*/receipt.json)
# into one sorted table, flagging rows whose tg delta exceeds the observed
# between-run card-state drift bound (±0.68%). Rows inside the bound are
# measurement state, not effects. Writes results/knob-ab-ledger.md (committed).
set -euo pipefail
cd "$(dirname "$0")/.."

DRIFT="${DRIFT:-0.68}" # % tg, observed between-run drift 20260806T063740Z vs 060605Z
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for d in $(ls -d results/ab-* 2>/dev/null | sort); do
  f="$d/receipt.json"
  [ -f "$f" ] || continue
  jq -r '[.stamp, (.candidate_env // .note // "?"), (.tg_delta_pct // 0), (.pp_delta_pct // 0), (.verdict // "?")] | @tsv' "$f" >> "$TMP" 2>/dev/null || true
done

out=results/knob-ab-ledger.md
{
  echo "# Knob A/B ledger — interleaved vs cooled source champion"
  echo ""
  echo "tg_delta_pct is candidate−champion decode delta in %; drift bound ±${DRIFT}% (between-run card-state)."
  echo "Rows outside the bound are potential real effects; all others are measurement state."
  echo ""
  echo "| stamp | candidate | tg_delta% | pp_delta% | verdict |"
  echo "|---|---|---|---|---|"
  sort "$TMP" | awk -F'\t' -v d="$DRIFT" '
    { tg=$3+0; b=(tg>=d||tg<=-d)?"**BEATS-DRIFT**":"-"; printf "| %s | %s | %s | %s | %s |\n", $1, $2, tg, $4+0, b }'
  echo ""
  echo "Rows beating the drift bound:"
  sort "$TMP" | awk -F'\t' -v d="$DRIFT" '($3+0>=d||$3+0<=-d){print "  " $1 " " $2 " tg=" $3 "%"}' | head -40
} > "$out"

echo "wrote $out ($(wc -l < "$out") lines)"
