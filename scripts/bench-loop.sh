#!/usr/bin/env bash
# bench-loop.sh — one clean candidate loop for the serial track (lx).
# Sources env.sh (quality-safe fuse stack), runs the golden gate, then the
# official serial bench. Prints a compact score receipt. Safe to run on a loop.
# Usage: ./scripts/bench-loop.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== env =="
source ./env.sh
env | rg '^(GGML_SYCL|ONEAPI_DEVICE_SELECTOR|ZE_AFFINITY_MASK)=' || true

echo "== golden smoke =="
./scripts/golden-smoke.sh

echo "== serial bench =="
./scripts/bench-serial.sh

echo "== latest score =="
jq '{score, increase_pct, decode_tok_s, prefill_tok_s, decode_speedup, prefill_speedup, floors_ok}' results/LATEST_SCORE.json

echo "== regression guard =="
./scripts/guard-score.sh
