#!/usr/bin/env bash
# mmid-sandwich.sh — same-window ctrl|combo|ctrl official-geometry sandwich to
# decide whether the MMID knob family (ENABLE_MMID_FUSED_SINGLE +
# MMID_WG_SUBGROUPS) is tg-neutral + pp-positive, or an artifact.
# Usage: bash scripts/mmid-sandwich.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/results/mmid-sandwich-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

export LX_LLAMA_BENCH="$ROOT/results/src-repro-20260806T035656Z/bin/llama-bench"
export LX_LLAMA_SERVER="$ROOT/results/src-repro-20260806T035656Z/bin/llama-server"

run_arm() { # $1=label; extra env already exported
  local label="$1" stamp
  echo "=== arm $label $(date -u +%H:%M:%SZ)" | tee -a "$OUT/ledger.md"
  stamp=$(bash "$ROOT/scripts/bench-serial.sh" --note "$label" 2>&1 | grep -oE 'results/[0-9]{8}T[0-9]{6}Z' | tail -1 | sed 's#results/##')
  echo "stamp=$stamp" | tee -a "$OUT/ledger.md"
  python3 - "$ROOT/results/$stamp/metrics.json" "$OUT/ledger.md" "$label" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
line = f"{sys.argv[3]}: tg={d['tg128']:.2f} pp={d['pp512']:.2f}"
print(line); open(sys.argv[2], "a").write(line + "\n")
PY
}

# ctrl-a
run_arm ctrl-a
# combo
export GGML_SYCL_ENABLE_MMID_FUSED_SINGLE=1
export GGML_SYCL_MMID_WG_SUBGROUPS=16
run_arm combo
# ctrl-b (unset knobs)
unset GGML_SYCL_ENABLE_MMID_FUSED_SINGLE GGML_SYCL_MMID_WG_SUBGROUPS
run_arm ctrl-b
echo "done: $OUT" | tee -a "$OUT/ledger.md"
