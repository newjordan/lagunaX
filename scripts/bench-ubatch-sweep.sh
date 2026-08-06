#!/usr/bin/env bash
# bench-ubatch-sweep.sh — sweep n_ubatch (prefill chunking) on the champion
# binary. Inverts a pinned assumption of the official geometry (-ub 2048):
# for a 512-token prompt the fused MoE down-GEMM dispatches as ONE 512-token
# GEMM (123-145 ms/call, ~16.6% of the pp512 dispatch span). Smaller -ub
# chunks that GEMM into 2x256 / 4x128 dispatches — the split-GEMM experiment
# (open lead 9) done at the RUNTIME level: no rebuild, no source edit.
# Quality: chunking changes fp reduction association in attention/ffn (the
# standard -ub path shipped in llama.cpp), so the winning arm gets a post-hoc
# greedy-smoke match before any claim (golden gate), mirroring the knobs.
# pp512-chunked prefill is PPL-suspect per env.sh's any-batch warning, so the
# official pp512 geometry result is diagnostic; a champion-lmhead tg128
# sandwich guard is run first for the decode side.
#
# Same-window ctrl sandwich: ub2048 runs FIRST and LAST; candidates between.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/results/ubatch-sweep-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

# --- env + lock harness (repo conventions: PREPEND, with-gpu-lock --wait) ----
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi
if [ -f "$ROOT/scripts/lib-gpu-lock.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/lib-gpu-lock.sh"
fi

# --- bench binary resolution: champion bin tree first, then PATH -------------
BENCH=""
for c in \
  "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench" \
  "${BINTREE:-}/llama-bench" "${LX_BIN:-}/llama-bench" \
  "$(command -v llama-bench 2>/dev/null || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then BENCH="$c"; break; fi
done
if [ -z "$BENCH" ]; then
  echo "FATAL: no llama-bench found (tried \$BINTREE, \$LX_BIN, PATH)" | tee "$OUT/fatal.txt"
  exit 1
fi
echo "bench: $BENCH" | tee "$OUT/bench.txt"

GEOM=(-m "${LX_MODEL:-}" -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ctk f16 -ctv f16 -r 3 -o json)
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_GRAPH=1
export GGML_SYCL_DISABLE_DNN=1

ARMS=(2048 1024 512 256 128 2048)   # ctrl-a, candidates, ctrl-b
IDX=0
for U in "${ARMS[@]}"; do
  if [ "$U" = 2048 ]; then TAG="ub2048-$IDX"; IDX=$((IDX+1)); else TAG="ub$U"; fi
  echo "=== arm $TAG (-ub $U) $(date -u +%H:%M:%SZ)" | tee -a "$OUT/ledger.md"
  if "$ROOT/scripts/with-gpu-lock" --wait 900 --reason "ubatch-$TAG" -- \
    env "$BENCH" "${GEOM[@]}" -ub "$U" > "$OUT/$TAG.json" 2> "$OUT/$TAG.stderr"; then
    rc=0
  else
    rc=$?
  fi
  echo "rc=$rc" >> "$OUT/$TAG.stderr"
  echo "rc=$rc" | tee -a "$OUT/ledger.md"
done

echo "=== summary ===" | tee -a "$OUT/ledger.md"
for f in "$OUT"/ub*.json; do
  [ -e "$f" ] || continue
  python3 - "$f" "$OUT/ledger.md" <<'PY'
import json, sys
f, led = sys.argv[1], sys.argv[2]
tag = f.split("/")[-1].replace(".json","")
try:
    with open(f) as fh:
        raw = "".join(
            ln for ln in fh
            if not ln.lstrip().startswith("[lx-") and not ln.lstrip().startswith("[layer-timer]")
        )
    for d in json.loads(raw):
        if isinstance(d, dict) and "avg_ts" in d:
            name = d.get("name", d.get("test", "?"))
            print(f"{tag} [{name}]: {d.get('avg_ts', float('nan'))} t/s (sd {d.get('stddev_ts','?')}, n {d.get('samples_ts','?')})")
            with open(led, "a") as out:
                out.write(f"{tag} [{name}]: {d.get('avg_ts')} t/s (sd {d.get('stddev_ts')})\n")
except Exception as e:
    print(f"{tag}: parse error {e}")
PY
done
echo "done: $OUT" | tee -a "$OUT/ledger.md"
