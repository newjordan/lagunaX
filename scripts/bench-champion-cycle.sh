#!/usr/bin/env bash
# bench-champion-cycle.sh — one-shot champion submission cycle.
#
# Purpose: close the gap in lead #1 (open leads, findings #2): ship the
# source-reproducible champion binary ONLY after both quality gates pass
# (golden-smoke + proof-suite), then run the official scoring bench and
# capture the receipt. Replaces the irreproducible binary-patched binary
# with a clean source build as the champion path.
#
# Loop contract:
#   gates (golden + proof)  -> abort on failure, champion NOT updated
#   bench-serial (official) -> results/<stamp>/score.json receipt
#   guard-score             -> floors / score bookkeeping
#
# Usage:
#   bash scripts/bench-champion-cycle.sh [--skip-proof] [--note "msg"]
#
# Exit codes:
#   0  scored and safe (gates passed)
#   75 gpu lock busy / concurrent job (same as bench-serial lock refusal)
#   76 golden-smoke failed (quality regression — do NOT ship)
#   77 proof-suite failed (regression — do NOT ship)
#   78 harness missing (scripts not found)

set -uo pipefail
cd "$(dirname "$0")/.." || exit 78

NOTE="${NOTE:-}"
SKIP_PROOF=0
# Binary under test — defaults to env.sh champion; pass --bin <dir> to gate a
# specific build (e.g. the source-repro build) instead of the binary patch.
CAND_BIN="${CAND_BIN:-}"
for arg in "$@"; do
  case "$arg" in
    --skip-proof) SKIP_PROOF=1 ;;
    --bin=*) CAND_BIN="${arg#--bin=}" ;;
    --bin) CAND_BIN="${2:-}"; shift ;;
    --note=*) NOTE="${arg#--note=}" ;;
    --note) NOTE="${2:-}"; shift ;;
  esac
done

if [[ -n "$CAND_BIN" ]]; then
  if [[ ! -x "$CAND_BIN/llama-bench" ]]; then
    echo "FATAL: --bin $CAND_BIN has no llama-bench" >&2
    exit 78
  fi
  export LX_BIN="$CAND_BIN"
  export LX_LLAMA_BENCH="$CAND_BIN/llama-bench"
  export LX_LLAMA_CLI="$CAND_BIN/llama-cli"
  export LX_LLAMA_SERVER="$CAND_BIN/llama-server"
  export LD_LIBRARY_PATH="$CAND_BIN:${LD_LIBRARY_PATH:-}"
  echo "== [cycle] candidate binary: $CAND_BIN"
else
  echo "== [cycle] candidate binary: env default (\$LX_BIN)"
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "FATAL: $1 not found" >&2; exit 78; }; }
need bash; need python3

[ -f scripts/golden-smoke.sh ] || { echo "FATAL: scripts/golden-smoke.sh missing" >&2; exit 78; }
[ -f scripts/bench-serial.sh ] || { echo "FATAL: scripts/bench-serial.sh missing" >&2; exit 78; }
[ -f scripts/guard-score.sh ] || { echo "FATAL: scripts/guard-score.sh missing" >&2; exit 78; }

echo "== [cycle] gates: golden-smoke"
if ! bash scripts/golden-smoke.sh; then
  echo "== [cycle] GOLDEN-SMOKE FAILED — quality regression, champion NOT updated" >&2
  exit 76
fi
echo "== [cycle] golden-smoke OK"

if [ "$SKIP_PROOF" -eq 0 ] && [ -f scripts/proof-suite.sh ]; then
  echo "== [cycle] gates: proof-suite"
  if ! bash scripts/proof-suite.sh; then
    echo "== [cycle] PROOF-SUITE FAILED — regression, champion NOT updated" >&2
    exit 77
  fi
  echo "== [cycle] proof-suite OK"
else
  echo "== [cycle] proof-suite skipped (not present or --skip-proof)"
fi

echo "== [cycle] official bench"
# Thermal-cooldown gate: the proof-suite leaves the B70 hot (49 min sustained
# load), and a bench run immediately after measures ~1.7% low (observed:
# tg 135.9 / pp 1145 right after proof vs tg 137.9 / pp 1163 after cooldown).
# Idle the card for a fixed window before the scored run.
COOLDOWN_S="${COOLDOWN_S:-360}"
TARGET_C="${TARGET_C:-60}"
MAX_COOLDOWN_S="${MAX_COOLDOWN_S:-1500}"
HWMON="${THERMAL_HWMON:-}"
if [ -z "$HWMON" ]; then
  for d in /sys/class/drm/card*/device/hwmon/hwmon*; do
    [ -d "$d" ] && HWMON="$d" && break
  done
fi
max_temp_mc() { # hottest VRAM-channel temp in millidegrees; empty if unreadable
  local f v max=0
  for f in "$HWMON"/temp*_input; do
    [ -r "$f" ] || continue
    v="$(cat "$f" 2>/dev/null || true)"
    [ -n "$v" ] && [ "$v" -gt "$max" ] 2>/dev/null && max="$v"
  done
  [ "$max" -gt 0 ] && echo "$max"
}
echo "== [cycle] cooldown min ${COOLDOWN_S}s then wait until max temp <= ${TARGET_C}C (cap ${MAX_COOLDOWN_S}s)"
sleep "$COOLDOWN_S"
if [ -n "$HWMON" ] && t="$(max_temp_mc)" && [ -n "$t" ]; then
  t0="$(date +%s)"
  while [ "$t" -gt $((TARGET_C * 1000)) ] && [ $(( $(date +%s) - t0 )) -lt "$MAX_COOLDOWN_S" ]; do
    echo "  [cycle] cooling: $((t / 1000)).$((t % 1000 / 100))C > ${TARGET_C}C — wait 30s"
    sleep 30
    t="$(max_temp_mc)"
  done
  echo "== [cycle] cooled: $((t / 1000)).$((t % 1000 / 100))C (target ${TARGET_C}C)"
fi
EXTRA_ARGS=()
if [ -n "$NOTE" ]; then EXTRA_ARGS+=(--note "$NOTE"); fi
if ! bash scripts/bench-serial.sh "${EXTRA_ARGS[@]}"; then
  rc=$?
  echo "== [cycle] bench-serial exit $rc" >&2
  exit "$rc"
fi

# latest scored receipt for the caller / polling
LATEST=$(ls -1dt results/*/score.json 2>/dev/null | head -1)
if [ -n "${LATEST:-}" ]; then
  echo "== [cycle] RECEIPT: $LATEST"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d,indent=1))' "$LATEST"
fi
echo "== [cycle] DONE"
