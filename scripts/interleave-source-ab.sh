#!/bin/bash
# interleave-source-ab.sh — same-window source-level A/B for the B70 champion.
# Each arm runs the official bench geometry (pp512/tg128, reps=5) through the
# existing bench-cold.sh path (gpu-lock + thermal gate + score pipeline), with
# an env-gated source selector (GGML_SYCL_LMHEAD_KPATH=dmmv|mmvq|mmq, compiled
# into the champion build tree by the probe mechanism) and interleaved CTRL
# arms. Deltas are judged against the ~0.7% between-run drift bound — a same-
# window CTRL sandwich kills the ambient confounder (open lead 12/24).
#
# Usage:
#   interleave-source-ab.sh [arm [arm ...]]
#   arm := label:value   (value "-" = unset/CTRL, else GGML_SYCL_LMHEAD_KPATH=value)
#   default: ctrl-a:- kp-dmmv:dmmv ctrl-b:- kp-mmvq:mmvq kp-mmq:mmq ctrl-c:-
#
# Resume semantics: any label already present in results/ (as a *complete* arm
# with a score) is skipped, so re-running after an interruption only benches
# the missing arms — the sweep is idempotent.
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
if [[ -f "$BUILD/probe-build.sh" ]]; then
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
  echo "[iab] rebuilt + relinked probe into $BUILD/bin/llama-bench"
else
  echo "[iab] no probe-build.sh; using existing $BUILD/bin/llama-bench"
fi

# 2) model + official bench geometry from env.sh
[ -f ./env.sh ] && source ./env.sh >/dev/null 2>&1 || true
MODEL=${MODEL:-${LX_MODEL:-}}
if [ -z "$MODEL" ]; then
  MODEL=$(find "$ROOT/baseline" -maxdepth 3 -name '*.gguf' 2>/dev/null | head -1)
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "no model"; exit 6; }
PP=${PP:-512}; TG=${TG:-128}

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/results/lmhead-kpath-$TS
mkdir -p "$OUT"
LEDGER=$OUT/ledger.tsv
: > "$LEDGER"
echo -e "arm\tvalue\ttg_ts\tpp_ts\tscore" >> "$LEDGER"

# default arm list: ctrl, dmmv, ctrl, mmvq, mmq, ctrl (sweep order)
if [ $# -eq 0 ]; then
  ARMS=(ctrl-a:- kp-dmmv:dmmv ctrl-b:- kp-mmvq:mmvq kp-mmq:mmq ctrl-c:-)
else
  ARMS=("$@")
fi

run_arm() { # $1=label:value
  local spec=$1 lbl=${1%%:*} kp=${1#*:}
  # idempotence: skip labels already scored in this run dir or the previous one
  local prev=$ROOT/results/lmhead-kpath-20260806T114031Z
  if grep -q "\"note\".*lmhead-kpath-$lbl" "$prev"/*.log "$prev"/../*/score.json 2>/dev/null; then
    echo "[iab] skip $lbl (already benched)"
    return 0
  fi
  echo "[iab] arm $lbl (kp=${kp:-unset}) — acquiring lock…"
  local kp_env=()
  if [ "$kp" != "-" ]; then kp_env=(GGML_SYCL_LMHEAD_KPATH="$kp"); fi
  env LX_MODEL="$MODEL" LX_PP="$PP" LX_TG="$TG" \
    LX_LLAMA_BENCH="$BUILD/bin/llama-bench" LX_BIN="$BUILD/bin" "${kp_env[@]}" \
    bash "$ROOT/scripts/bench-cold.sh" --note "lmhead-kpath-$lbl" \
    >"$OUT/$lbl.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[iab] arm $lbl FAILED rc=$rc"; return $rc; fi
  local tg pp sc
  tg=$(grep -oE '"tg128": [0-9.]+' "$OUT/$lbl.log" | head -1 | grep -oE '[0-9.]+')
  pp=$(grep -oE '"pp512": [0-9.]+' "$OUT/$lbl.log" | head -1 | grep -oE '[0-9.]+')
  sc=$(grep -oE '"score": [0-9.]+' "$OUT/$lbl.log" | head -1 | grep -oE '[0-9.]+')
  echo -e "$lbl\t${kp:-unset}\t${tg:-NA}\t${pp:-NA}\t${sc:-NA}" >> "$LEDGER"
  echo "[iab] $lbl tg=$tg pp=$pp score=$sc"
}

for spec in "${ARMS[@]}"; do
  run_arm "$spec" || true
done
echo "[iab] done — ledger: $LEDGER"
cat "$LEDGER"
