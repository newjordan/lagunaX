#!/usr/bin/env bash
# bench-knob-candidate.sh — golden-gated runtime-knob candidate bench.
#
# The champion source reads its MoE/fusion control surface from runtime env
# knobs (ggml-sycl.cpp getenv), so a candidate is an ENV DIFF on the shipped
# binary — no rebuild. This runner: waits for the GPU lock to free, runs
# golden-smoke with the knob(s) ON, and only if greedy still matches does it
# run the official bench (scored receipt). Both outcomes are recorded so a
# knob is either shipped or killed with evidence.
#
# Usage:
#   KNOB_SPEC="GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1" \
#     NOTE="mmid-fused-batch" \
#     bash scripts/bench-knob-candidate.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

NOTE="${NOTE:-knob-candidate}"
KNOB_SPEC="${KNOB_SPEC:-}"
MAX_MIN="${MAX_MIN:-480}"
DEADLINE=$(( $(date +%s) + MAX_MIN*60 ))
LOG="results/knob-${NOTE}-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "== [knob] $NOTE : $KNOB_SPEC — waiting for GPU lock (deadline +${MAX_MIN}m)"

# Wait until no llama process holds the card (bench-when-free style), then
# let bench-serial's own lock acquire under us. Never kill anything.
while pgrep -f 'llama-(server|bench|cli)' >/dev/null 2>&1; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "== [knob] give up after ${MAX_MIN}m — GPU never free"; exit 3
  fi
  sleep 60
done
echo "== [knob] GPU free at $(date -u +%H:%M:%SZ)"

if [ -n "$KNOB_SPEC" ]; then
  export "$KNOB_SPEC"
  echo "== [knob] exported $KNOB_SPEC"
fi

echo "== [knob] golden-smoke (greedy gate) with $KNOB_SPEC"
if ! bash scripts/golden-smoke.sh; then
  echo "== [knob] GOLDEN FAIL — killing knob $KNOB_SPEC (quality gate)"; exit 2
fi
echo "== [knob] GOLDEN OK — running official bench"
bash scripts/bench-serial.sh --note "$NOTE:${KNOB_SPEC:-baseline-flags}"
RC=$?
echo "== [knob] bench exit=$RC (log: $LOG)"
exit $RC
