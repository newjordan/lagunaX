#!/bin/bash
# lmhead-prefetch-cycle.sh — the only unburned lever on the B70 decode path:
# a DEFAULT-OFF (champion-bitexact) source edit to the fused q6_K lm_head GEMV
# inside src-lmhead/ggml/src/ggml-sycl/mmvq.cpp.
#
# Measured context (see results/lmhead-probe-ledger.md, results/layer-timer/):
#   - fused lm_head group (mul_mat+add+add2, wtype=q6_K, l_out-<L>) = 353.6 us/token
#   - ~4.8% of the ~7.3 ms decode iteration; elimination ceiling +5.12% tg
#   - effective BW ~475 GB/s vs ~2 TB/s card capability  => per-block dequant
#     /DRAM-latency bound, not ALU or BW bound (VDR 2/4/8 all null, kpath null,
#     ~40 GGML_SYCL_* knobs null-or-negative).
#   - the ONE untouched axis: load-order of q6_K blocks (software pipelining /
#     next-block prefetch). Pure load reordering is bit-exact: same bytes, same
#     accumulate order.
#
# The patch payload lives OUTSIDE this script at results/lmhead-prefetch/q6k.patch
# so its anchors are authored against the real mmvq.cpp, not guessed here. This
# harness validates the payload (patch --dry-run, idempotence guard), then runs
# the repo's proven cycle: compile probe object -> swap into src-lmhead-build ->
# relink via CMake link.txt -> golden-smoke WITH the gate ON -> same-window
# CTRL vs candidate sandwich -> ledger row. Any anchor drift aborts rc=20
# BEFORE touching source. Env gate GGML_SYCL_LMHEAD_PREFETCH defaults to 0,
# so the shipped champion path is byte-identical.
#
# Usage: bash scripts/lmhead-prefetch-cycle.sh
#   SKIP_COMPILE=1            reuse existing probe object (edit-only iteration)
#   DRY_RUN=1                 validate patch + env plumbing, do not build/bench
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=$ROOT/src-lmhead
BUILD=$ROOT/src-lmhead-build
PATCH_DIR=$ROOT/results/lmhead-prefetch
PATCH=$PATCH_DIR/q6k.patch
GATE=GGML_SYCL_LMHEAD_PREFETCH

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib
[ -d /opt/intel/oneapi/compiler/2026.0/bin ] && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2026.0/bin

echo "[lmhead-prefetch] gate=$GATE (default OFF = champion bitexact)"

# --- (0) validate the patch payload -------------------------------------
if [ ! -f "$PATCH" ]; then
  echo "[lmhead-prefetch] MISSING payload: $PATCH (author against the real mmvq.cpp first)"; exit 20
fi
if grep -q "$GATE" "$PATCH"; then :; else
  echo "[lmhead-prefetch] payload does not reference $GATE — refusing"; exit 20
fi
if [ -d "$SRC/.git" ]; then
  ( cd "$SRC" && git apply --check "$PATCH" ) >/tmp/lmhead-prefetch-check.log 2>&1 \
    || { echo "[lmhead-prefetch] patch does not apply cleanly (anchor drift?):"; cat /tmp/lmhead-prefetch-check.log; exit 20; }
else
  # patch fallback: --forward = never prompt on already-applied/reversed hunks
  ( cd "$SRC" && patch --dry-run --forward -p1 < "$PATCH" ) >/tmp/lmhead-prefetch-check.log 2>&1 \
    || { echo "[lmhead-prefetch] patch --dry-run failed (anchor drift?):"; cat /tmp/lmhead-prefetch-check.log; exit 20; }
fi
echo "[lmhead-prefetch] payload validated against live mmvq.cpp"

# idempotence guard: refuse to double-apply (checked BEFORE validation so a
# previously-applied payload fails fast instead of hanging patch's -R prompt)
if grep -q "$GATE" "$SRC/ggml/src/ggml-sycl/mmvq.cpp"; then
  echo "[lmhead-prefetch] gate already present in mmvq.cpp — refusing re-apply (git apply -R ../results/lmhead-prefetch/q6k.patch first)"; exit 20
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[lmhead-prefetch] DRY_RUN=1: payload + env plumbing validated, stopping before build/bench"; exit 0
fi

# --- (1) apply payload ----------------------------------------------------
( cd "$SRC" && git apply "$PATCH" 2>/dev/null || patch --forward -p1 < "$PATCH" ) \
  || { echo "[lmhead-prefetch] APPLY FAIL"; exit 21; }
# revert via git apply -R (reverse) — NEVER git checkout: this worktree's index
# lives under the write-denied base repo and checkout dies on index.lock.
trap 'cd "$SRC" && ( git apply -R "$PATCH" || patch -R -p1 < "$PATCH" ) >/dev/null 2>&1 || true' EXIT
echo "[lmhead-prefetch] applied $PATCH"

# --- (2) compile probe object, swap, relink (mirrors lmhead-probe-cycle) ---
[ -f "$BUILD/probe-build.sh" ] || { echo "missing $BUILD/probe-build.sh"; exit 2; }
[ -x "$BUILD/bin/llama-bench" ] || { echo "missing champion llama-bench"; exit 2; }
if [ -z "${SKIP_COMPILE:-}" ] || [ ! -f "$BUILD/probe-ggml-sycl.o" ]; then
  bash "$BUILD/probe-build.sh" >/tmp/lmhead-prefetch-build.log 2>&1 \
    || { echo "BUILD FAIL:"; tail -20 /tmp/lmhead-prefetch-build.log; exit 3; }
fi
OBJ=$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)
if [ -z "$OBJ" ]; then
  touch "$SRC/ggml/src/ggml-sycl/ggml-sycl.cpp"
  cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/lmhead-prefetch-link.log 2>&1 \
    || { echo "CMAKE BUILD FAIL:"; tail -20 /tmp/lmhead-prefetch-link.log; exit 4; }
else
  cp "$BUILD/probe-ggml-sycl.o" "$OBJ"
  LINK=$(dirname "$OBJ")/link.txt
  if [ -f "$LINK" ]; then
    ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >/tmp/lmhead-prefetch-link.log 2>&1 \
      || { echo "RELINK FAIL:"; tail -20 /tmp/lmhead-prefetch-link.log; exit 5; }
  else
    cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/lmhead-prefetch-link.log 2>&1 \
      || { echo "CMAKE RELINK FAIL:"; tail -20 /tmp/lmhead-prefetch-link.log; exit 5; }
  fi
fi
echo "[lmhead-prefetch] probe built, swapped, relinked"

# --- (3) golden-smoke WITH the gate ON (quality-invisibility gate) ---------
if [ "${GOLDEN:-1}" = "1" ]; then
  GATE="$GATE" bash scripts/golden-smoke.sh >/tmp/lmhead-prefetch-golden.log 2>&1 \
    || { echo "GOLDEN FAIL (gate not quality-neutral):"; tail -20 /tmp/lmhead-prefetch-golden.log; exit 6; }
  echo "[lmhead-prefetch] GOLDEN OK"
fi

# --- (4) same-window CTRL vs candidate sandwich via the proven loop ---------
echo "[lmhead-prefetch] arm1 CTRL (gate off) -> arm2 candidate (gate on) -> arm3 CTRL"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "results/lmhead-prefetch-$STAMP"
# arm1: CTRL
./scripts/with-gpu-lock --wait -- env -u "$GATE" "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m /mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf \
  > "results/lmhead-prefetch-$STAMP/ctrl-a.log" 2>&1 || { echo "CTRL-A FAIL"; exit 7; }
# arm2: candidate
./scripts/with-gpu-lock --wait -- env "$GATE"=1 "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m /mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf \
  > "results/lmhead-prefetch-$STAMP/cand.log" 2>&1 || { echo "CAND FAIL"; exit 7; }
# arm3: CTRL
./scripts/with-gpu-lock --wait -- env -u "$GATE" "$BUILD/bin/llama-bench" \
  -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -r 5 -o json -m /mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf \
  > "results/lmhead-prefetch-$STAMP/ctrl-b.log" 2>&1 || { echo "CTRL-B FAIL"; exit 7; }

# --- (5) ledger row ---------------------------------------------------------
python3 - "$STAMP" <<'PY'
import json, sys
stamp = sys.argv[1]
def objs(p):
    s = open(f"results/lmhead-prefetch-{stamp}/{p}").read()
    # drop stderr control lines that interleave with the JSON stream
    s = "\n".join(l for l in s.split("\n")
                  if not l.startswith("[lx-") and not l.startswith("[lmhead") and not l.startswith("[l-gpu"))
    out=[]; i=0
    while True:
        i=s.find("{", i)
        if i<0: break
        dep=0; done=False
        for k in range(i,len(s)):
            c=s[k]
            if c=="{": dep+=1
            elif c=="}":
                dep-=1
                if dep==0:
                    try: out.append(json.loads(s[i:k+1]))
                    except Exception: pass
                    i=k+1; done=True
                    break
        if not done: break
    return out
def tg(p):
    for o in objs(p):
        if o.get("n_gen") == 128:   # official-geometry decode test object
            return o["avg_ts"]
    raise SystemExit(f"no tg object in {p}")
a, c, b = tg("ctrl-a.log"), tg("cand.log"), tg("ctrl-b.log")
mean = (a + b) / 2
delta = (c - mean) / mean * 100
row = f"| {stamp} | {a:.3f} | {c:.3f} | {b:.3f} | {delta:+.3f}% | {mean:.3f} |\n"
with open("results/lmhead-prefetch-LEDGER.md", "a") as f:
    f.write(row)
print(f"[lmhead-prefetch] ctrl-a={a:.3f} cand={c:.3f} ctrl-b={b:.3f} delta_vs_ctrl_mean={delta:+.3f}%")
PY
echo "[lmhead-prefetch] done (board untouched; promote only if |delta|>0.68% and positive)"
