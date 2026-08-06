#!/usr/bin/env bash
# bench-when-free.sh — wait for the B70 GPU lock to be free, then run the
# official golden gate + serial bench loop. Does NOT kill anything.
# Usage: ./scripts/bench-when-free.sh [max_minutes]
set -uo pipefail
cd "$(dirname "$0")/.."
MAX_MIN=${1:-600}
DEADLINE=$(( $(date +%s) + MAX_MIN*60 ))

while pgrep -f 'llama-(server|bench|cli)' >/dev/null 2>&1; do
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
