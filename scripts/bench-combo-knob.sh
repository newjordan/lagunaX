#!/usr/bin/env bash
# bench-combo-knob.sh — combo-runtime-knob bench cycle on the ACTIVE champion binary.
#
# Structural difference from prior knob work: instead of rebuilding or diffing
# source, it flips runtime-only GGML_SYCL_* control knobs (pure getenv reads,
# e.g. ggml-sycl.cpp GGML_SYCL_ENABLE_MMID_FUSED_BATCH:5483 /
# GGML_SYCL_DISABLE_MOE_PACKED_REDUCE:6222) and benches the SAME binary already
# recorded as the board champion. Every candidate is golden-gated before any
# bench time is spent. Source reads are GGML_SYCL_-prefixed only; unprefixed
# names are silently ignored, so only GGML_SYCL_* counts as a knob.
#
# Usage: GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1 scripts/bench-combo-knob.sh
#        (export any GGML_SYCL_* knobs; golden-gated, then official bench)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1) Resolve the active champion binary from the scored receipt (not a guess).
LATEST="$ROOT/results/LATEST_SCORE.json"
if [[ ! -f "$LATEST" ]]; then
  echo "no LATEST_SCORE.json — run a scored bench first" >&2; exit 2
fi
BINARY="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["candidate_meta"]["binary"])' "$LATEST")"
[[ -x "$BINARY" ]] || { echo "champion binary missing: $BINARY" >&2; exit 2; }
SERVER="${BINARY%/llama-bench}/llama-server"
[[ -x "$SERVER" ]] || { echo "paired llama-server missing: $SERVER" >&2; exit 2; }

# 2) Exported knob surface (caller-controlled). Bench child inherits these;
#    env.sh uses :- defaults for LX_* so it will not clobber the GGML_SYCL_* names.
KNOB_DIFF="$(env | grep -E '^GGML_SYCL_[A-Z0-9_]+=' || true)"
LABEL="combo$(echo "$KNOB_DIFF" | sed 's/^GGML_SYCL_//;s/^[A-Z_]*=//;s/=$//' | tr '\n' ',')"
[[ -n "$KNOB_DIFF" ]] || { echo "no GGML_SYCL_* knob env set — refusing to bench a no-op candidate" >&2; exit 2; }

# 3) Golden gate on the same binary with the same env (greedy smoke match).
#    Wait out the exclusive GPU lock (proof-suite/other benches own the card):
#    the card is single-client, and golden+bench must be serial.
DEADLINE=$(( $(date +%s) + 4*3600 ))
echo "== golden-smoke (env: ${KNOB_DIFF//$'\n'/ }) — lock wait until +4h =="
while true; do
  bash "$ROOT/scripts/golden-smoke.sh" >"$ROOT/results/golden-combo-last.log" 2>&1
  GOLDEN_RC=$?
  if [[ $GOLDEN_RC -eq 0 ]]; then break; fi
  if [[ $GOLDEN_RC -eq 75 ]]; then
    if [[ $(date +%s) -ge $DEADLINE ]]; then
      echo "lock deadline hit — giving up" >&2; exit 5
    fi
    echo "$(date -u +%H:%M:%S) card busy (lock rc=75) — retry in 90s"
    sleep 90
  else
    echo "GOLDEN FAIL rc=$GOLDEN_RC — candidate killed before bench" >&2
    tail -5 "$ROOT/results/golden-combo-last.log" >&2
    exit 3
  fi
done
echo "GOLDEN OK"

# 4) Official serial bench (env.sh binary default == champion; lock exits 75
#    if the card is held; bench-serial writes the scored receipt itself).
export LX_LLAMA_BENCH="$BINARY"
export LX_LLAMA_SERVER="$SERVER"
OUT="$(cd "$ROOT" && bash scripts/bench-serial.sh --note "$LABEL" 2>&1)"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "bench-serial rc=$RC (75 = card busy — rerun later)" >&2
  echo "$OUT" | tail -15 >&2
  exit $RC
fi

# 5) Surface the scored receipt.
STAMP="$(echo "$OUT" | grep -oE 'results/[0-9]{8}T[0-9]{6}Z' | tail -1 | sed 's#results/##')"
if [[ -z "$STAMP" ]]; then
  STAMP="$(ls -1dt "$ROOT"/results/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z 2>/dev/null | head -1 | xargs -r basename)"
fi
echo "stamp=$STAMP"
if [[ -f "$ROOT/results/$STAMP/score.json" ]]; then
  python3 - "$ROOT/results/$STAMP/score.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"score={d['score']:.6f} inc={d['increase_pct']:+.2f}% tg={d['decode_tok_s']:.2f} pp={d['prefill_tok_s']:.2f}")
PY
else
  echo "no score.json at $STAMP (floors failed?) — check $ROOT/results/$STAMP/metrics.json" >&2
  exit 4
fi
