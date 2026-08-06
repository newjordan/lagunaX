#!/usr/bin/env bash
# bench-thermal-gated-cycle.sh — thermal-gated champion submission cycle.
#
# NEW DIRECTION vs bench-champion-cycle.sh: bench-champion-cycle benches the
# moment gates finish; but the proof-suite sustains 50 min of load and the
# card then under-reports by ~2% (evidence: 1.1908 hot right after proof vs
# 1.2181 for the SAME binary+flags after >=5 min idle — results/20260806T050115Z
# vs results/20260806T060605Z). This cycle adds a temperature/cooldown gate
# BEFORE the official bench so every receipt is a cooled, reproducible score.
#
# Pipeline:
#   gates (golden-smoke + proof-suite)  -> abort on fail (76/77)
#   cooldown gate (temp < TARGET, or COOLDOWN_S min idle)  -> thermal.json sidecar
#   bench-champion-cycle (--skip-proof) -> golden re-check + official bench +
#                                          guard-score + board + commit (0/75/76/78)
#
# Usage:
#   bash scripts/bench-thermal-gated-cycle.sh [--bin <dir>] [--skip-proof]
#        [--cooldown-s 600] [--target-c 55] [--max-wait-s 1800] [--note "msg"]
#
# Exit codes:
#   0  scored and safe (gates passed, cooled bench submitted)
#   75 gpu lock busy / concurrent job
#   76 golden-smoke failed
#   77 proof-suite failed
#   78 harness missing / bad args

set -uo pipefail
cd "$(dirname "$0")/.." || exit 78

SKIP_PROOF=0
CAND_BIN=""
COOLDOWN_S=600
TARGET_C=62
MAX_WAIT_S=1800
NOTE="thermal-gated-cycle"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-proof) SKIP_PROOF=1 ;;
    --bin) CAND_BIN="${2:-}"; shift ;;
    --bin=*) CAND_BIN="${1#--bin=}" ;;
    --cooldown-s) COOLDOWN_S="${2:-600}"; shift ;;
    --target-c) TARGET_C="${2:-62}"; shift ;;
    --max-wait-s) MAX_WAIT_S="${2:-1800}"; shift ;;
    --note) NOTE="${2:-thermal-gated-cycle}"; shift ;;
    --note=*) NOTE="${1#--note=}" ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 78 ;;
  esac
  shift
done

for s in golden-smoke proof-suite bench-champion-cycle; do
  [ -f "scripts/$s.sh" ] || { echo "FATAL: scripts/$s.sh missing" >&2; exit 78; }
done

# --- gates ----------------------------------------------------------------
echo "== [tgate] gates: golden-smoke"
if ! bash scripts/golden-smoke.sh; then
  echo "== [tgate] GOLDEN-SMOKE FAILED — champion NOT updated" >&2
  exit 76
fi
echo "== [tgate] golden-smoke OK"

if [ "$SKIP_PROOF" -eq 0 ]; then
  echo "== [tgate] gates: proof-suite (this heats the card — cooldown follows)"
  if ! bash scripts/proof-suite.sh; then
    echo "== [tgate] PROOF-SUITE FAILED — champion NOT updated" >&2
    exit 77
  fi
  echo "== [tgate] proof-suite OK"
else
  echo "== [tgate] proof-suite skipped (--skip-proof)"
fi

# --- cooldown gate ----------------------------------------------------------
# B70 xe hwmon exposes per-VRAM-channel temps (temp6..temp19_input) but NOT
# temp1_input; the max channel temp tracks card thermal state (the 2.3% score
# swing after the 50-min proof-suite). Aggregate the max; fall back to pure
# idle sleep when no sensor exists.
temp_c() {
  local max_v=0 found=0 v f
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input; do
    [ -r "$f" ] || continue
    v=$(cat "$f" 2>/dev/null) || continue
    [ -n "$v" ] || continue
    v=$(( v / 1000 ))
    found=1
    [ "$v" -gt "$max_v" ] && max_v=$v
  done
  if [ "$found" -eq 1 ]; then echo "$max_v"; return 0; fi
  return 1
}

STAMP="thermal-$(date -u +%Y%m%dT%H%M%SZ)"
THERMAL_LOG="results/$STAMP-thermal.json"
mkdir -p results

ELAPSED=0
TEMP=""
STABLE_START=0
{
  echo "{\"stamp\":\"$STAMP\",\"target_c\":$TARGET_C,\"cooldown_s\":$COOLDOWN_S,\"probe\":\"max-vram-channel-c\",\"readings\":["
  FIRST=1
  while [ "$ELAPSED" -lt "$MAX_WAIT_S" ]; do
    T="$(temp_c || echo '')"
    if [ -n "$T" ]; then
      [ "$FIRST" -eq 1 ] || echo ","
      FIRST=0
      printf '{"t_s":%d,"temp_c":%d}' "$ELAPSED" "$T"
      if [ "$T" -le "$TARGET_C" ]; then
        if [ "$STABLE_START" -eq 0 ]; then STABLE_START=$ELAPSED; fi
        if [ $(( ELAPSED - STABLE_START )) -ge 120 ]; then
          TEMP=$T
          echo ""
          break
        fi
      else
        STABLE_START=0
      fi
    fi
    if [ -z "$TEMP" ] && [ "$ELAPSED" -ge "$COOLDOWN_S" ]; then
      # no temp source or target unreachable within cooldown — idle budget spent
      TEMP="${T:-unknown}"
      echo ""
      break
    fi
    ELAPSED=$(( ELAPSED + 15 ))
    sleep 15
  done
  echo "],\"final_temp_c\":\"$TEMP\",\"idle_s\":$ELAPSED,\"note\":\"$NOTE\"}"
} > "$THERMAL_LOG" 2>&1

echo "== [tgate] cooldown complete: idle ${ELAPSED}s, final temp ${TEMP}C (log: $THERMAL_LOG)"

# --- official bench + board + commit (golden re-check inside) --------------
ARGS=(--skip-proof --note="$NOTE")
if [ -n "$CAND_BIN" ]; then ARGS+=(--bin="$CAND_BIN"); fi
echo "== [tgate] official bench (champion-cycle, cooled)"
if ! bash scripts/bench-champion-cycle.sh "${ARGS[@]}"; then
  rc=$?
  echo "== [tgate] bench-champion-cycle exited $rc" >&2
  exit "$rc"
fi
echo "== [tgate] SUBMITTED cooled receipt (thermal log: $THERMAL_LOG)"
exit 0
