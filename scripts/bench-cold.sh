#!/usr/bin/env bash
# bench-cold.sh — official bench gated on a cold card (thermal precondition).
#
# New submission path: instead of benching whenever the cycle fires, wait
# until the B70's hwmon temp is below THERMAL_THRESHOLD_C, then run the
# existing official bench (bench-serial.sh, which owns the GPU lock and the
# scored receipt). This removes the ~2.3% heat-depression from the pipeline
# so the board reflects the binary, not the card's thermal history.
#
# Exit codes: thermal-gate failures (2=no hwmon, 1=timeout) abort WITHOUT
# benching; anything from bench-serial.sh is passed through.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! bash scripts/thermal-gate.sh; then
  echo "bench-cold: ABORT — card not cold, refusing to bench (score would be heat-depressed)" >&2
  exit 3
fi

echo "bench-cold: card cold, running official bench" >&2
exec bash scripts/bench-serial.sh "$@"
