#!/usr/bin/env bash
# Serial-only bench: pp512 + tg128 → results/<stamp>/metrics.json (+ score if baseline exists)
# Usage:
#   source env.sh && ./scripts/bench-serial.sh --baseline   # pin baseline/
#   source env.sh && ./scripts/bench-serial.sh [--note "text"]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

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

# Exclusive B70: concurrent Level-Zero clients wedge xe (xe_validation_lock).
lx_gpu_lock_enter "bench-serial${NOTE:+:$NOTE}" || exit $?
trap 'lx_gpu_lock_leave' EXIT

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
BENCH_SHA256="$(sha256sum "$LX_LLAMA_BENCH" | awk '{print $1}')"

# Keep disabled as the true llama-bench default; pass a value only for explicit trials.
NUMA_ARGS=()
if [[ -n "$LX_NUMA" ]]; then
  case "$LX_NUMA" in
    distribute|isolate|numactl) NUMA_ARGS=(--numa "$LX_NUMA") ;;
    *) echo "invalid LX_NUMA: $LX_NUMA (expected empty, distribute, isolate, or numactl)" >&2; exit 2 ;;
  esac
fi

case "$LX_SPLIT_MODE" in
  none|layer|row|tensor) ;;
  *) echo "invalid LX_SPLIT_MODE: $LX_SPLIT_MODE (expected none, layer, row, or tensor)" >&2; exit 2 ;;
esac
[[ "$LX_MAIN_GPU" =~ ^[0-9]+$ ]] || { echo "invalid LX_MAIN_GPU: $LX_MAIN_GPU" >&2; exit 2; }
[[ "$LX_TENSOR_SPLIT" =~ ^[0-9]+([.][0-9]+)?(/[0-9]+([.][0-9]+)?)*$ ]] || { echo "invalid LX_TENSOR_SPLIT: $LX_TENSOR_SPLIT" >&2; exit 2; }
[[ -n "$LX_DEVICE" && "$LX_DEVICE" != *[[:space:]]* ]] || { echo "invalid LX_DEVICE: $LX_DEVICE (expected auto or a slash-separated device list without whitespace)" >&2; exit 2; }
[[ "$LX_THREADS_BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "invalid LX_THREADS_BATCH: $LX_THREADS_BATCH (expected a positive integer)" >&2; exit 2; }
[[ "$LX_NO_KV_OFFLOAD" =~ ^[01]$ ]] || { echo "invalid LX_NO_KV_OFFLOAD: $LX_NO_KV_OFFLOAD (expected 0 or 1)" >&2; exit 2; }
[[ "$LX_CPU_MOE_LAYERS" =~ ^[0-9]+$ ]] || { echo "invalid LX_CPU_MOE_LAYERS: $LX_CPU_MOE_LAYERS (expected a non-negative integer)" >&2; exit 2; }
[[ "$LX_NO_OP_OFFLOAD" =~ ^[01]$ ]] || { echo "invalid LX_NO_OP_OFFLOAD: $LX_NO_OP_OFFLOAD (expected 0 or 1)" >&2; exit 2; }
[[ "$LX_NO_HOST" =~ ^[01]$ ]] || { echo "invalid LX_NO_HOST: $LX_NO_HOST (expected 0 or 1)" >&2; exit 2; }
[[ "$LX_NO_WARMUP" =~ ^[01]$ ]] || { echo "invalid LX_NO_WARMUP: $LX_NO_WARMUP (expected 0 or 1)" >&2; exit 2; }
[[ "$LX_SYCL_DISABLE_GRAPH" =~ ^[01]$ ]] || { echo "invalid LX_SYCL_DISABLE_GRAPH: $LX_SYCL_DISABLE_GRAPH (expected 0 or 1)" >&2; exit 2; }
[[ -z "$LX_SYCL_DISABLE_DNN" || "$LX_SYCL_DISABLE_DNN" =~ ^[01]$ ]] || { echo "invalid LX_SYCL_DISABLE_DNN: $LX_SYCL_DISABLE_DNN (expected empty, 0, or 1)" >&2; exit 2; }
[[ "$LX_DELAY" =~ ^[0-9]+$ ]] || { echo "invalid LX_DELAY: $LX_DELAY (expected a non-negative integer number of seconds)" >&2; exit 2; }
[[ "$LX_FIT_TARGET_MIB" =~ ^[0-9]+$ ]] || { echo "invalid LX_FIT_TARGET_MIB: $LX_FIT_TARGET_MIB (expected a non-negative integer MiB value)" >&2; exit 2; }
[[ "$LX_FIT_CTX" =~ ^[1-9][0-9]*$ ]] || { echo "invalid LX_FIT_CTX: $LX_FIT_CTX (expected a positive integer)" >&2; exit 2; }
[[ "$LX_COMBINED_BENCH" =~ ^[01]$ ]] || { echo "invalid LX_COMBINED_BENCH: $LX_COMBINED_BENCH (expected 0 or 1)" >&2; exit 2; }
[[ "$LX_DEPTH" =~ ^[0-9]+$ ]] || { echo "invalid LX_DEPTH: $LX_DEPTH (expected a non-negative integer)" >&2; exit 2; }
WARMUP_ARGS=()
[[ "$LX_NO_WARMUP" == 1 ]] && WARMUP_ARGS=(--no-warmup)
FIT_ARGS=()
[[ "$LX_FIT_TARGET_MIB" != 0 ]] && FIT_ARGS=(--fit-target "$LX_FIT_TARGET_MIB" --fit-ctx "$LX_FIT_CTX")
# This llama-bench has no --threads-batch (prefill folds into -t); skip it.
# FA: only emit when explicitly on/off/auto; FA=-1 means "use binary default"
# (auto → FA on via the FA-VEC-GQA patch), which matches the pinned champion run.
FA_ARGS=()
case "$FA" in
  on|off|auto) FA_ARGS=(-fa "$FA") ;;
esac

# Note: this llama-bench build has no -c/--ctx; window size is -p/-n only.
COMMON=(
  -m "$LX_MODEL"
  -ngl "$NGL"
    --n-cpu-moe "$LX_CPU_MOE_LAYERS"
    --split-mode "$LX_SPLIT_MODE"
    --main-gpu "$LX_MAIN_GPU"
    --tensor-split "$LX_TENSOR_SPLIT"
    --device "$LX_DEVICE"
  -t "$THREADS"
  --cpu-mask "$LX_CPU_MASK"
  --cpu-strict "$LX_CPU_STRICT"
  -b "$BBATCH"
  -ub "$UBATCH"
  -ctk "$CTK"
  -ctv "$CTV"
  --no-kv-offload "$LX_NO_KV_OFFLOAD"
    --no-op-offload "$LX_NO_OP_OFFLOAD"
      --no-host "$LX_NO_HOST"
  "${FA_ARGS[@]}"
  -r "$LX_REPS"
  -d "$LX_DEPTH"
  --prio "$LX_PRIO"
  --load-mode "$LX_LOAD_MODE"
  --poll "$LX_POLL"
  --delay "$LX_DELAY"
  "${FIT_ARGS[@]}"
  "${WARMUP_ARGS[@]}"
  "${NUMA_ARGS[@]}"
  -o json
)

echo "== lx serial bench =="
echo "  mode:    $MODE"
echo "  binary:  $LX_LLAMA_BENCH"
echo "  bin sha: $BENCH_SHA256"
echo "  model:   $LX_MODEL"
  echo "  window:  pp${LX_PP} / tg${LX_TG}  reps=$LX_REPS combined_bench=$LX_COMBINED_BENCH delay=$LX_DELAY warmup_disabled=$LX_NO_WARMUP threads=$THREADS threads_batch=$LX_THREADS_BATCH cpu_moe_layers=$LX_CPU_MOE_LAYERS no_kv_offload=$LX_NO_KV_OFFLOAD no_op_offload=$LX_NO_OP_OFFLOAD no_host=$LX_NO_HOST sycl_disable_graph=$LX_SYCL_DISABLE_GRAPH sycl_disable_dnn=${LX_SYCL_DISABLE_DNN:-backend-default} priority=$LX_PRIO load=$LX_LOAD_MODE poll=$LX_POLL numa=${LX_NUMA:-disabled} cpu_mask=$LX_CPU_MASK cpu_strict=$LX_CPU_STRICT split=$LX_SPLIT_MODE main_gpu=$LX_MAIN_GPU tensor_split=$LX_TENSOR_SPLIT device=$LX_DEVICE"
echo "  device:  $ONEAPI_DEVICE_SELECTOR  ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo "  out:     $METRICS_JSON"

# A combined run avoids loading and initializing the same model twice. Keep the
# legacy two-process path available for direct comparison and compatibility.
if [[ "$LX_COMBINED_BENCH" == 1 ]]; then
  echo "-- pp${LX_PP} + tg${LX_TG} (combined) --" | tee "$RAW_LOG"
  COMBINED_JSON="$("$LX_LLAMA_BENCH" "${COMMON[@]}" -pg "${LX_PP},${LX_TG}" 2>>"$RAW_LOG")" || {
    echo "combined pp/tg bench failed" >&2
    exit 1
  }
  echo "$COMBINED_JSON" >>"$RAW_LOG"
  PP_JSON="$COMBINED_JSON"
  TG_JSON="$COMBINED_JSON"
else
  echo "-- pp${LX_PP} --" | tee "$RAW_LOG"
  PP_JSON="$("$LX_LLAMA_BENCH" "${COMMON[@]}" -p "$LX_PP" -n 0 2>>"$RAW_LOG")" || {
    echo "pp bench failed" >&2
    exit 1
  }
  echo "$PP_JSON" >>"$RAW_LOG"

  echo "-- tg${LX_TG} --" | tee -a "$RAW_LOG"
  TG_JSON="$("$LX_LLAMA_BENCH" "${COMMON[@]}" -p 0 -n "$LX_TG" 2>>"$RAW_LOG")" || {
    echo "tg bench failed" >&2
    exit 1
  }
  echo "$TG_JSON" >>"$RAW_LOG"
fi

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

# Retain per-rep samples + stddev (llama-bench -o json carries them; the legacy
# mean-only parse dropped them). Lets score.py report measurement noise so a
# candidate win is distinguishable from run variance.
parse_meta() {
  local blob="$1" want="$2"
  python3 - "$blob" "$want" <<'PY'
import json, sys
blob, want = sys.argv[1], sys.argv[2]
data = json.loads(blob)
rows = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [data])
for r in rows:
    name = str(r.get("test") or r.get("name") or "")
    n_prompt = r.get("n_prompt", r.get("ps", None))
    n_gen = r.get("n_gen", r.get("tg", None))
    if "avg_ts" not in r:
        continue
    if want.startswith("pp") and (name.startswith("pp") or (n_gen in (0, "0", None) and n_prompt)):
        pass
    elif want.startswith("tg") and (name.startswith("tg") or (n_prompt in (0, "0", None) and n_gen)):
        pass
    elif len(rows) == 1:
        pass
    else:
        continue
    print(json.dumps({"stddev_ts": r.get("stddev_ts"), "samples_ts": r.get("samples_ts")}))
    sys.exit(0)
print("row not found", file=sys.stderr)
sys.exit(1)
PY
}
PP_META="$(parse_meta "$PP_JSON" "pp${LX_PP}")" || PP_META='{}'
TG_META="$(parse_meta "$TG_JSON" "tg${LX_TG}")" || TG_META='{}'

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
    "pp_samples": $PP_META,
    "tg_samples": $TG_META,
    "window": {"pp": int("$LX_PP"), "tg": int("$LX_TG"), "reps": int("$LX_REPS"), "depth": int("$LX_DEPTH")},
      "combined_bench": bool(int("$LX_COMBINED_BENCH")),
    "binary": "$LX_LLAMA_BENCH",
    "binary_sha256": "$BENCH_SHA256",
    "model": "$LX_MODEL",
    "flags": {
        "ngl": int("$NGL"),
        "threads": int("$THREADS"),
          "threads_batch": int("$LX_THREADS_BATCH"),
          "cpu_moe_layers": int("$LX_CPU_MOE_LAYERS"),
        "cpu_mask": "$LX_CPU_MASK",
        "cpu_strict": int("$LX_CPU_STRICT"),
          "split_mode": "$LX_SPLIT_MODE",
          "main_gpu": int("$LX_MAIN_GPU"),
          "tensor_split": "$LX_TENSOR_SPLIT",
            "device": "$LX_DEVICE",
        "ubatch": int("$UBATCH"),
        "bbatch": int("$BBATCH"),
        "ctx": int("$CTX"),
        "ctk": "$CTK",
        "ctv": "$CTV",
        "no_kv_offload": int("$LX_NO_KV_OFFLOAD"),
          "no_op_offload": int("$LX_NO_OP_OFFLOAD"),
          "no_host": int("$LX_NO_HOST"),
        "flash_attn": int("$FA"),
        "sycl_disable_graph": int("$LX_SYCL_DISABLE_GRAPH"),
          "sycl_disable_dnn": "$LX_SYCL_DISABLE_DNN" or "backend-default",
        "priority": int("$LX_PRIO"),
        "load_mode": "$LX_LOAD_MODE",
        "poll": int("$LX_POLL"),
        "delay_seconds": int("$LX_DELAY"),
        "numa": "$LX_NUMA" or "disabled",
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
  # also update LATEST pointers (monotonic board: never downgrade below the max verified score)
  echo "$OUT_DIR" >"$LX_RESULTS/LATEST_DIR.txt"
  # Promotion is not "score went up". A candidate may legitimately change its
  # own runtime config, but changing the config changes what the two speedup
  # terms mean — on 2026-08-06 a ub=2048 change took the board with prefill
  # 1.02 -> 2.97 while decode went 1.2932 -> 1.2851, i.e. the 0.75-weight term
  # regressed and the board still moved. So:
  #   same measurement geometry -> a plain score win promotes
  #   changed geometry          -> decode must ALSO not regress
  #   either way                -> the KLD quality gate must have passed
  # scripts/promote-gate.py owns the decision so every caller shares it.
  python3 "$ROOT/scripts/promote-gate.py" \
    --candidate "$SCORE_JSON" \
    --board "$LX_RESULTS/LATEST_SCORE.json" \
    --kld "$LX_RESULTS/LATEST_KLD.json" \
    --rejected "$LX_RESULTS/LATEST_SCORE_REJECTED.json"
  echo "score → $SCORE_JSON"
fi

if [[ "$MODE" == "baseline" ]]; then
  echo "BASELINE PINNED → $METRICS_JSON"
  echo "Do not re-run --baseline unless intentionally re-contracting."
fi
