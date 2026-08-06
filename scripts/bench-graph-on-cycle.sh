#!/usr/bin/env bash
# bench-graph-on-cycle.sh — official-pipeline A/B of the last untouched runtime
# knob on the SYCL backend: command-list graph capture (GGML_SYCL_DISABLE_GRAPH).
# The champion env pins DISABLE_GRAPH=1 (env.sh:76); every other SYCL control
# surface (VDR, MMID, packed-reduce, dual-down, LMHEAD_Q8, fa, ubatch, ...) has
# been measured; the graph enable/disable axis has never had a same-window A/B
# on this build. This cycle runs it through the official golden→bench-serial→
# score/guard pipeline exactly like bench-knob-candidate.sh, so the result is
# either a shipped improvement or a killed knob with evidence.
#
# Usage:  bash scripts/bench-graph-on-cycle.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NOTE="graph-enable"
MAX_MIN="${MAX_MIN:-240}"
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
LOG="results/graph-enable-$(date +%Y%m%dT%H%M%SZ).log"
mkdir -p results

exec > >(tee -a "$LOG") 2>&1
echo "== [graph] cycle start $(date -u +%FT%TZ)"

# Wait until no llama process actually holds the card (CPU-only embedding
# daemon with -ngl 0 must NOT block the queue — lib-gpu-lock.sh predicate).
source "$ROOT/scripts/lib-gpu-lock.sh"
llama_gpu_busy() {
  local pid
  for pid in $(pgrep -f 'llama-(server|bench|cli)' 2>/dev/null); do
    if ls -l "/proc/$pid/fd" 2>/dev/null | grep -q '/dev/dri'; then
      return 0
    fi
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- '-ngl 0'; then
      return 0
    fi
  done
  return 1
}
while llama_gpu_busy; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "== [graph] give up after ${MAX_MIN}m — GPU never free"; exit 3
  fi
  sleep 60
done
echo "== [graph] GPU free at $(date -u +%H:%M:%SZ)"

# Flip the knob: env.sh defaults LX_SYCL_DISABLE_GRAPH=1; export the override
# BEFORE bench-serial sources env.sh so ${LX_SYCL_DISABLE_GRAPH:-1} keeps it.
export LX_SYCL_DISABLE_GRAPH=0
export GGML_SYCL_DISABLE_GRAPH=0
echo "== [graph] exported GGML_SYCL_DISABLE_GRAPH=0 (graph capture ENABLED)"

echo "== [graph] golden-smoke (greedy quality gate) with graph enabled"
if ! bash scripts/golden-smoke.sh; then
  echo "== [graph] GOLDEN FAIL — killing knob (quality regression)"; exit 2
fi
echo "== [graph] GOLDEN OK — running official bench (pp512/tg128, r=5, scored)"
bash scripts/bench-serial.sh --note "$NOTE"
RC=$?
echo "== [graph] bench exit=$RC (log: $LOG)"
exit $RC
