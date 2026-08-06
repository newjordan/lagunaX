#!/bin/bash
# bench-lmhead-kpath.sh — A/B the lm_head decode GEMV kernel path on the B70.
# Source-level env-gated selector in src-lmhead/ggml/src/ggml-sycl/ggml-sycl.cpp
# (GGML_SYCL_LMHEAD_KPATH=dmmv|mmvq|mmq), compiled into the champion build tree
# via the existing probe mechanism. Each arm: official bench geometry
# (pp512/tg128) with that kernel path forced for the fused lm_head group ONLY;
# all other layers keep the shipped path. Interleaved CTRL (unset = shipped
# default selection) and candidate arms in one lock window, deltas judged
# against the ~0.7% between-run drift bound.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=$ROOT/src-lmhead
BUILD=$ROOT/src-lmhead-build

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib
[ -d /opt/intel/oneapi/compiler/2026.0/bin ] && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2026.0/bin

# 1) rebuild probe object from the edited source and relink into champion binary
bash "$BUILD/probe-build.sh" >/tmp/kpath-build.log 2>&1 || { echo "BUILD FAIL"; tail -20 /tmp/kpath-build.log; exit 2; }
OBJ=$(find "$BUILD" -name 'ggml-sycl.cpp.o' -path '*ggml-sycl*' | head -1)
[ -n "$OBJ" ] || { echo "no ggml-sycl.cpp.o in build tree"; exit 3; }
cp "$BUILD/probe-ggml-sycl.o" "$OBJ"
LINK=$(dirname "$OBJ")/link.txt
if [ -f "$LINK" ]; then
  ( cd "$(dirname "$(dirname "$(dirname "$LINK")")")" && bash "$LINK" ) >/tmp/kpath-link.log 2>&1 \
    || { echo "RELINK FAIL"; tail -20 /tmp/kpath-link.log; exit 4; }
else
  cmake --build "$BUILD" --target ggml-sycl -j32 >/tmp/kpath-link.log 2>&1 || { echo "CMAKE FAIL"; tail -20 /tmp/kpath-link.log; exit 4; }
fi
[ -x "$BUILD/bin/llama-bench" ] || { echo "llama-bench missing after relink"; exit 5; }
echo "[kpath] rebuilt + relinked probe into $(readlink -f "$BUILD/bin/llama-bench" || echo $BUILD/bin/llama-bench)"

# 2) model + official bench geometry from env.sh
[ -f ./env.sh ] && source ./env.sh >/dev/null 2>&1 || true
MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  # env.sh exports LX_MODEL (the official laguna model); only fall back to a
  # search when env.sh itself didn't define it.
  MODEL=${LX_MODEL:-}
fi
if [ -z "$MODEL" ]; then
  MODEL=$(find "$ROOT/baseline" -maxdepth 3 -name '*.gguf' 2>/dev/null | head -1)
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "no model"; exit 6; }
PP=${PP:-512}; TG=${TG:-128}
echo "[kpath] model=$MODEL pp$PP/tg$TG"

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/results/lmhead-kpath-$TS
mkdir -p "$OUT"

run_arm() { # $1=label $2=kpath-value (or - for unset)
  local lbl=$1 kp=$2
  echo "[kpath] arm $lbl (kp=${kp:-unset}) — acquiring lock…"
  local kp_env=()
  if [ "$kp" != "-" ]; then kp_env=(GGML_SYCL_LMHEAD_KPATH="$kp"); fi
  # CRITICAL: force bench-serial to use the RELINKED probe binary — env.sh's
  # default LX_BIN points at the stock champion worktree, which would make
  # every arm (and the kpath env) a silent no-op.
  env LX_MODEL="$MODEL" LX_PP="$PP" LX_TG="$TG" \
    LX_LLAMA_BENCH="$BUILD/bin/llama-bench" LX_BIN="$BUILD/bin" "${kp_env[@]}" \
    bash "$ROOT/scripts/bench-cold.sh" --note "lmhead-kpath-$lbl" \
    >"$OUT/$lbl.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[kpath] arm $lbl FAILED rc=$rc"; return $rc; fi
  grep -E '"avg_ts"|"pp512"|"tg128"|"score"' "$OUT/$lbl.log" | tail -5
}

run_arm ctrl-a -
run_arm kp-dmmv dmmv
run_arm ctrl-b -
run_arm kp-mmvq mmvq
run_arm ctrl-c -
run_arm kp-mmq mmq
run_arm ctrl-d -

echo "[kpath] done → $OUT"
