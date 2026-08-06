#!/usr/bin/env bash
# bench-when-free.sh — wait for the B70 GPU lock to be free, then run the
# official golden gate + serial bench loop. Does NOT kill anything.
# Usage: ./scripts/bench-when-free.sh [max_minutes]
set -uo pipefail
cd "$(dirname "$0")/.."
MAX_MIN=${1:-600}
DEADLINE=$(( $(date +%s) + MAX_MIN*60 ))

# Busy only if a llama process actually holds the GPU: CPU-only servers
# (-ngl 0, e.g. the embedding daemon on :8091) must not block the queue.
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
    echo "[bench-when-free] give up after ${MAX_MIN}m — GPU still busy" >&2
    exit 3
  fi
  echo "[bench-when-free] $(date -u +%H:%M:%SZ) GPU busy, waiting 60s..."
  sleep 60
done
echo "[bench-when-free] GPU free — starting official loop"
exec bash scripts/bench-loop.sh
