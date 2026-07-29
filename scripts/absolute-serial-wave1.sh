#!/usr/bin/env bash
# Wave 1: serial absolute-limit knob ladder on B70.
# Each arm → results/abs-serial-w1/<arm>/ + board JSON.
# Does NOT re-pin baseline; scores vs baseline/baseline.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BOARD_DIR="$LX_RESULTS/abs-serial-w1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BOARD_DIR"
BOARD_JSON="$BOARD_DIR/BOARD.json"
echo "board → $BOARD_DIR"

BENCH="$LX_LLAMA_BENCH"
MODEL="$LX_MODEL"
REPS="${LX_REPS:-5}"

run_arm() {
  local name="$1"
  shift
  # remaining: env KEY=VAL pairs, then -- then llama-bench extra flags
  local -a envpairs=()
  local -a flags=()
  local phase=env
  for a in "$@"; do
    if [[ "$a" == "--" ]]; then phase=flags; continue; fi
    if [[ "$phase" == env ]]; then envpairs+=("$a"); else flags+=("$a"); fi
  done

  local adir="$BOARD_DIR/$name"
  mkdir -p "$adir"
  echo
  echo "======== ARM $name ========"
  echo "  env: ${envpairs[*]:-(none)}"
  echo "  flags: ${flags[*]:-(defaults)}"

  # clean slate of SYCL knobs we might touch
  unset GGML_SYCL_DISABLE_DNN GGML_SYCL_DISABLE_FUSION GGML_SYCL_ENABLE_FUSION
  unset GGML_SYCL_DISABLE_GRAPH GGML_SYCL_GRAPH
  unset GGML_SYCL_DISABLE_TOPK_MOE GGML_SYCL_FORCE_MMQ
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  export ZE_AFFINITY_MASK=0
  export LD_LIBRARY_PATH="$LX_BIN:${LD_LIBRARY_PATH:-}"

  for kv in "${envpairs[@]+"${envpairs[@]}"}"; do
    [[ -z "${kv:-}" ]] && continue
    export "$kv"
  done

  local common=(
    -m "$MODEL"
    -ngl 99
    -ctk f16
    -ctv f16
    -r "$REPS"
    -o json
  )

  # defaults if arm doesn't set -ub/-b/-fa/-t (avoid multi-value accidental pairs)
  local has_ub=0 has_b=0 has_fa=0 has_t=0
  for f in "${flags[@]+"${flags[@]}"}"; do
    [[ "$f" == "-ub" || "$f" == "--ubatch-size" ]] && has_ub=1
    [[ "$f" == "-b" || "$f" == "--batch-size" ]] && has_b=1
    [[ "$f" == "-fa" || "$f" == "--flash-attn" ]] && has_fa=1
    [[ "$f" == "-t" || "$f" == "--threads" ]] && has_t=1
  done
  (( has_ub == 0 )) && flags+=(-ub 2048)
  (( has_b == 0 )) && flags+=(-b 2048)
  (( has_fa == 0 )) && flags+=(-fa on)
  (( has_t == 0 )) && flags+=(-t 16)

  local raw="$adir/bench.log"
  : >"$raw"

  echo "-- pp512 --" | tee -a "$raw"
  local pp_json
  if ! pp_json="$("$BENCH" "${common[@]}" "${flags[@]}" -p 512 -n 0 2>>"$raw")"; then
    echo "FAIL pp $name" | tee -a "$raw"
    echo "{\"name\":\"$name\",\"ok\":false,\"phase\":\"pp\"}" >"$adir/metrics.json"
    return 0
  fi
  echo "$pp_json" >>"$raw"

  echo "-- tg128 --" | tee -a "$raw"
  local tg_json
  if ! tg_json="$("$BENCH" "${common[@]}" "${flags[@]}" -p 0 -n 128 2>>"$raw")"; then
    echo "FAIL tg $name" | tee -a "$raw"
    echo "{\"name\":\"$name\",\"ok\":false,\"phase\":\"tg\"}" >"$adir/metrics.json"
    return 0
  fi
  echo "$tg_json" >>"$raw"

  python3 - "$adir/metrics.json" "$name" "$pp_json" "$tg_json" <<'PY'
import json, sys, os
from pathlib import Path
out, name, pp_blob, tg_blob = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def avg_ts(blob, kind):
    data = json.loads(blob)
    rows = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [data])
    for r in rows:
        avg = r.get("avg_ts")
        if avg is None:
            continue
        n_gen = r.get("n_gen")
        n_prompt = r.get("n_prompt")
        test = str(r.get("test") or "")
        if kind == "pp" and (test.startswith("pp") or n_gen in (0, "0", None)):
            return float(avg)
        if kind == "tg" and (test.startswith("tg") or n_prompt in (0, "0", None)):
            return float(avg)
    if len(rows) == 1 and rows[0].get("avg_ts") is not None:
        return float(rows[0]["avg_ts"])
    raise SystemExit(f"parse fail {kind}: {rows!r}"[:400])

pp = avg_ts(pp_blob, "pp")
tg = avg_ts(tg_blob, "tg")
payload = {
    "name": name,
    "ok": True,
    "stamp": os.environ.get("STAMP", ""),
    "track": "serial-absolute",
    "pp512": pp,
    "tg128": tg,
    "env": {
        "GGML_SYCL_DISABLE_DNN": os.environ.get("GGML_SYCL_DISABLE_DNN"),
        "GGML_SYCL_DISABLE_FUSION": os.environ.get("GGML_SYCL_DISABLE_FUSION"),
        "GGML_SYCL_DISABLE_GRAPH": os.environ.get("GGML_SYCL_DISABLE_GRAPH"),
        "GGML_SYCL_DISABLE_TOPK_MOE": os.environ.get("GGML_SYCL_DISABLE_TOPK_MOE"),
        "GGML_SYCL_FORCE_MMQ": os.environ.get("GGML_SYCL_FORCE_MMQ"),
    },
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(f"  → pp512={pp:.2f}  tg128={tg:.2f}")
PY

  # score vs pinned baseline
  if [[ -f "$LX_BASELINE_JSON" ]]; then
    python3 "$ROOT/scripts/score.py" \
      --baseline "$LX_BASELINE_JSON" \
      --candidate "$adir/metrics.json" \
      -o "$adir/score.json" 2>&1 | tee -a "$raw" || true
  fi
}

# --- arms ---
# 0. reconfirm baseline shape
run_arm reconfirm_baseline -- -ub 2048 -b 2048 -fa on

# 1. validated B70 prefill ship (ub/b up)
run_arm ship_ub4k_b8k -- -ub 4096 -b 8192 -fa on

# 2. oneDNN ON explicitly (unset is default; force clear)
run_arm dnn_on_ub4k -- -ub 4096 -b 8192 -fa on

# 3. native GEMM
run_arm dnn_off_ub4k GGML_SYCL_DISABLE_DNN=1 -- -ub 4096 -b 8192 -fa on

# 4. SYCL graph ON (launch overhead)
run_arm graph_on_ub4k GGML_SYCL_DISABLE_GRAPH=0 -- -ub 4096 -b 8192 -fa on

# 5. fusion OFF (expect decode loss)
run_arm fusion_off_ub4k GGML_SYCL_DISABLE_FUSION=1 -- -ub 4096 -b 8192 -fa on

# 6. topk moe off (expect decode loss on MoE)
run_arm topk_off_ub4k GGML_SYCL_DISABLE_TOPK_MOE=1 -- -ub 4096 -b 8192 -fa on

# 7. force MMQ
run_arm mmq_ub4k GGML_SYCL_FORCE_MMQ=1 -- -ub 4096 -b 8192 -fa on

# 8. bigger ub only / mid
run_arm ub512_b2k -- -ub 512 -b 2048 -fa on
run_arm ub1k_b4k -- -ub 1024 -b 4096 -fa on
run_arm ub2k_b4k -- -ub 2048 -b 4096 -fa on
run_arm ub4k_b4k -- -ub 4096 -b 4096 -fa on
run_arm ub8k_b8k -- -ub 8192 -b 8192 -fa on

# 9. FA off
run_arm fa_off_ub4k -- -ub 4096 -b 8192 -fa off

# 10. threads
run_arm t8_ub4k -- -ub 4096 -b 8192 -fa on -t 8
run_arm t32_ub4k -- -ub 4096 -b 8192 -fa on -t 32

# Assemble board
python3 - "$BOARD_DIR" "$BOARD_JSON" "$LX_BASELINE_JSON" <<'PY'
import json, sys
from pathlib import Path
board_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
base = json.loads(Path(sys.argv[3]).read_text()) if Path(sys.argv[3]).exists() else {}
rows = []
for d in sorted(board_dir.iterdir()):
    if not d.is_dir():
        continue
    m = d / "metrics.json"
    if not m.exists():
        continue
    mj = json.loads(m.read_text())
    sj = {}
    sp = d / "score.json"
    if sp.exists():
        sj = json.loads(sp.read_text())
    rows.append({
        "arm": d.name,
        "ok": mj.get("ok"),
        "pp512": mj.get("pp512"),
        "tg128": mj.get("tg128"),
        "score": sj.get("score"),
        "increase_pct": sj.get("increase_pct"),
        "decode_speedup": sj.get("decode_speedup"),
        "prefill_speedup": sj.get("prefill_speedup"),
    })

def key(r):
    s = r.get("score")
    return s if s is not None else -1

rows_sorted = sorted(rows, key=key, reverse=True)
best = rows_sorted[0] if rows_sorted else None
# also rank by raw tg and pp
by_tg = sorted([r for r in rows if r.get("tg128")], key=lambda r: r["tg128"], reverse=True)
by_pp = sorted([r for r in rows if r.get("pp512")], key=lambda r: r["pp512"], reverse=True)
payload = {
    "track": "serial-absolute-wave1",
    "baseline": {"pp512": base.get("pp512"), "tg128": base.get("tg128")},
    "best_score": best,
    "best_tg": by_tg[0] if by_tg else None,
    "best_pp": by_pp[0] if by_pp else None,
    "arms": rows_sorted,
}
out.write_text(json.dumps(payload, indent=2) + "\n")
print("\n===== BOARD (by score) =====")
print(f"{'arm':28} {'pp512':>10} {'tg128':>10} {'score':>8} {'incr%':>8}")
for r in rows_sorted:
    print(f"{r['arm'][:28]:28} {r.get('pp512') or 0:10.2f} {r.get('tg128') or 0:10.2f} "
          f"{(r.get('score') or 0):8.4f} {(r.get('increase_pct') or 0):+8.2f}")
print(f"\nbest score: {best}")
print(f"best tg:    {by_tg[0] if by_tg else None}")
print(f"best pp:    {by_pp[0] if by_pp else None}")
print(f"wrote {out}")
PY

echo "$BOARD_DIR" >"$LX_RESULTS/LATEST_ABS_SERIAL_DIR.txt"
echo "DONE wave1 → $BOARD_DIR"
