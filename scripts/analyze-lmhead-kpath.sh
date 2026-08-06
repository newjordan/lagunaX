#!/bin/bash
# analyze-lmhead-kpath.sh — summarize one lmhead-kpath A/B window: pull
# pp/tg from each arm's bench-serial receipt and compare candidate arms
# against the bracketing ctrl arms (drift-bounded).
# Usage: analyze-lmhead-kpath.sh results/lmhead-kpath-<TS>
set -euo pipefail
OUT=${1:?need lmhead-kpath-<TS> dir}
cd "$(dirname "$0")/.."
DIR=$ROOT 2>/dev/null || true
ROOT=$PWD
[ -d "$ROOT/$OUT" ] || OUT_DIR="$OUT"; OUT_DIR="$ROOT/$OUT"

echo "== lmhead-kpath window: $OUT =="
arm() { # $1=label
  local f="$OUT_DIR/$1.log"
  [ -f "$f" ] || { echo "  $1: MISSING ($f)"; return 1; }
  local pp tg score
  pp=$(grep -m1 '"pp512"' "$f" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  tg=$(grep -m1 '"tg128"' "$f" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  score=$(grep -m1 '"score"' "$f" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "  $1: pp512=$pp tg128=$tg score=$score"
}
arm ctrl-a || true
arm kp-dmmv || true
arm ctrl-b || true
arm kp-mmvq || true
arm ctrl-c || true
arm kp-mmq || true
arm ctrl-d || true
