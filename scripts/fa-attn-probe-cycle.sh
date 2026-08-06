#!/bin/bash
# fa-attn-probe-cycle.sh — probe the flash-attention axis on the SYCL backend.
# The official champion env carries GGML_SYCL_DISABLE_DNN=1; oneDNN is the SYCL FA
# implementation, so this sweeps -fa auto/on/off in official geometry, golden-gated.
# Pattern copied from vdrN-cycle.sh (lock wrapper, LD_LIBRARY_PATH PREPEND).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/results/fa-attn-$STAMP"
mkdir -p "$OUT"
BENCH="$ROOT/results/src-repro-20260806T035656Z/bin/llama-bench"
MODEL="/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf"
LOCK="$ROOT/scripts/with-gpu-lock"

run_arm() {
  local fa="$1" tag="$2"
  echo "=== arm $tag (-fa $fa) ===" | tee -a "$OUT/cycle.log"
  env -i HOME="$HOME" PATH="$PATH" TERM="$TERM" \
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 \
    GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1 \
    "$LOCK" --wait -- bash -c "
      source $ROOT/env.sh >/dev/null 2>&1
      export LD_LIBRARY_PATH=$ROOT/src-lmhead-build/bin:\${LD_LIBRARY_PATH:-}
      $BENCH -m $MODEL -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto \
        -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa $fa -r 5 -o json \
        > $OUT/$tag.json 2> $OUT/$tag.stderr
      echo \"arm $tag rc=\$?\"
    " 2>&1 | tail -3 | tee -a "$OUT/cycle.log"
}

for arm in "auto ctrl-auto" "on fa-on" "off fa-off"; do
  set -- $arm
  run_arm "$1" "$2"
done

echo "=== receipts ===" | tee -a "$OUT/cycle.log"
for f in ctrl-auto fa-on fa-off; do
  echo "-- $f: $(grep -o '\"avg_ts\":[0-9.]*' "$OUT/$f.json" 2>/dev/null | tr '\n' ' ')" | tee -a "$OUT/cycle.log"
done
