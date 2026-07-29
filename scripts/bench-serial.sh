#!/usr/bin/env bash
# Serial-only bench: pp512 + tg128 → results/<stamp>/metrics.json (+ score if baseline exists)
# Usage:
#   source env.sh && ./scripts/bench-serial.sh --baseline   # pin baseline/
#   source env.sh && ./scripts/bench-serial.sh [--note "text"]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

MODE="candidate"
NOTE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) MODE="baseline"; shift ;;
    --note) NOTE="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,6p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$LX_LLAMA_BENCH" ]]; then
  echo "missing llama-bench: $LX_LLAMA_BENCH" >&2
  exit 1
fi
if [[ ! -f "$LX_MODEL" ]]; then
  echo "missing model: $LX_MODEL" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$MODE" == "baseline" ]]; then
  OUT_DIR="$LX_ROOT/baseline"
  mkdir -p "$OUT_DIR"
  METRICS_JSON="$OUT_DIR/baseline.json"
  RAW_LOG="$OUT_DIR/baseline-raw-${STAMP}.log"
else
  OUT_DIR="$LX_RESULTS/$STAMP"
  mkdir -p "$OUT_DIR"
  METRICS_JSON="$OUT_DIR/metrics.json"
  RAW_LOG="$OUT_DIR/llama-bench.log"
fi

# Note: this llama-bench build has no -c/--ctx; window size is -p/-n only.
COMMON=(
  -m "$LX_MODEL"
  -ngl "$NGL"
  -t "$THREADS"
  -b "$BBATCH"
  -ub "$UBATCH"
  -ctk "$CTK"
  -ctv "$CTV"
  -r "$LX_REPS"
  -o json
)

echo "== lx serial bench =="
echo "  mode:    $MODE"
echo "  binary:  $LX_LLAMA_BENCH"
echo "  model:   $LX_MODEL"
echo "  window:  pp${LX_PP} / tg${LX_TG}  reps=$LX_REPS"
echo "  device:  $ONEAPI_DEVICE_SELECTOR  ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo "  out:     $METRICS_JSON"

# Prefill: n=0, p=512
echo "-- pp${LX_PP} --" | tee "$RAW_LOG"
PP_JSON="$("$LX_LLAMA_BENCH" "${COMMON[@]}" -p "$LX_PP" -n 0 2>>"$RAW_LOG")" || {
  echo "pp bench failed" >&2
  exit 1
}
echo "$PP_JSON" >>"$RAW_LOG"

# Decode: n=128, p small (0) — pure tg
echo "-- tg${LX_TG} --" | tee -a "$RAW_LOG"
TG_JSON="$("$LX_LLAMA_BENCH" "${COMMON[@]}" -p 0 -n "$LX_TG" 2>>"$RAW_LOG")" || {
  echo "tg bench failed" >&2
  exit 1
}
echo "$TG_JSON" >>"$RAW_LOG"

# Parse mean avg_ts from llama-bench JSON (array of rows)
parse_ts() {
  local blob="$1" test_name="$2"
  python3 - "$blob" "$test_name" <<'PY'
import json, sys
blob, want = sys.argv[1], sys.argv[2]
data = json.loads(blob)
if isinstance(data, dict) and "results" in data:
    rows = data["results"]
elif isinstance(data, list):
    rows = data
else:
    rows = [data]
for r in rows:
    # llama-bench fields vary slightly by version
    name = str(r.get("test") or r.get("name") or "")
    n_prompt = r.get("n_prompt", r.get("ps", None))
    n_gen = r.get("n_gen", r.get("tg", None))
    avg = r.get("avg_ts") or r.get("avg_tokens_per_second")
    if avg is None:
        continue
    if want.startswith("pp") and (name.startswith("pp") or (n_gen in (0, "0", None) and n_prompt)):
        print(float(avg))
        sys.exit(0)
    if want.startswith("tg") and (name.startswith("tg") or (n_prompt in (0, "0", None) and n_gen)):
        print(float(avg))
        sys.exit(0)
# fallback: single row
if len(rows) == 1 and rows[0].get("avg_ts") is not None:
    print(float(rows[0]["avg_ts"]))
    sys.exit(0)
print("parse failed; rows=", json.dumps(rows)[:500], file=sys.stderr)
sys.exit(1)
PY
}

PP_TS="$(parse_ts "$PP_JSON" "pp${LX_PP}")"
TG_TS="$(parse_ts "$TG_JSON" "tg${LX_TG}")"

python3 - "$METRICS_JSON" <<PY
import json, os, sys
from pathlib import Path
out = Path(sys.argv[1])
payload = {
    "stamp": os.environ.get("STAMP", "$STAMP"),
    "track": "serial",
    "mode": "$MODE",
    "pp512": float("$PP_TS"),
    "tg128": float("$TG_TS"),
    "window": {"pp": int("$LX_PP"), "tg": int("$LX_TG"), "reps": int("$LX_REPS")},
    "binary": "$LX_LLAMA_BENCH",
    "model": "$LX_MODEL",
    "flags": {
        "ngl": int("$NGL"),
        "threads": int("$THREADS"),
        "ubatch": int("$UBATCH"),
        "bbatch": int("$BBATCH"),
        "ctx": int("$CTX"),
        "ctk": "$CTK",
        "ctv": "$CTV",
    },
    "env": {
        "ONEAPI_DEVICE_SELECTOR": os.environ.get("ONEAPI_DEVICE_SELECTOR"),
        "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
        "GGML_SYCL_DISABLE_GRAPH": os.environ.get("GGML_SYCL_DISABLE_GRAPH"),
        "GGML_SYCL_DISABLE_DNN": os.environ.get("GGML_SYCL_DISABLE_DNN"),
    },
    "note": """$NOTE""",
    "claim_boundary": [
        "Serial only: one stream pp512 + tg128.",
        "Not multi-slot aggregate tok/s.",
        "Not comparable to mlx.fast M5 absolute tok/s (different quant/silicon).",
        "Score is vs pinned B70 baseline only.",
    ],
}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo
echo "wrote $METRICS_JSON"
echo "  pp${LX_PP}=${PP_TS}  tg${LX_TG}=${TG_TS}"

if [[ "$MODE" == "candidate" ]]; then
  if [[ ! -f "$LX_BASELINE_JSON" ]]; then
    echo "no baseline at $LX_BASELINE_JSON — run with --baseline first" >&2
    exit 0
  fi
  SCORE_JSON="$OUT_DIR/score.json"
  python3 "$ROOT/scripts/score.py" \
    --baseline "$LX_BASELINE_JSON" \
    --candidate "$METRICS_JSON" \
    -o "$SCORE_JSON"
  # also update LATEST pointers
  echo "$OUT_DIR" >"$LX_RESULTS/LATEST_DIR.txt"
  cp -f "$SCORE_JSON" "$LX_RESULTS/LATEST_SCORE.json"
  echo "score → $SCORE_JSON"
fi

if [[ "$MODE" == "baseline" ]]; then
  echo "BASELINE PINNED → $METRICS_JSON"
  echo "Do not re-run --baseline unless intentionally re-contracting."
fi
