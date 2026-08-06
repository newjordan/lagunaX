#!/usr/bin/env bash
# Laguna base (all custom fuses OFF) vs quality-safe tip (env.sh kills only).
# Same binary + same GGUF. No Treebeard/Qwen comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$LX_RESULTS/laguna-ab-$STAMP}"
BD="${BD:-$LX_BIN}"
MODEL="${MODEL:-$LX_MODEL}"
PORT="${PORT:-18940}"
BASE="http://127.0.0.1:${PORT}"
TEMPLATE="${LX_TEMPLATE:-/home/frosty40/turbo/worktrees/treebeard-pr-private-latest/models/templates/poolside-Laguna-XS-2.1.jinja}"
TOOL_ROOT="${TOOL_ROOT:-/tmp/tool-eval-bench}"
TOOL_BIN="$TOOL_ROOT/.venv/bin/tool-eval-bench"
HELD_OUT="${HELD_OUT:-/home/frosty40/turbo/held-out-probe}"
LONGCTX_PY="${LONGCTX_PY:-/home/frosty40/turbo/treebeard-work/research/treebeard-pr-private/run-long-context-eval.py}"
# Max practical for longctx/agent on B70+Laguna; tool schemas need ~35k+
SRV_CTX="${SRV_CTX:-32768}"
# Agent69: same ctx (64k OOM on this box)
AGENT_CTX="${AGENT_CTX:-32768}"

lx_gpu_lock_enter "laguna-ab-suite" || exit $?
trap 'lx_gpu_lock_leave' EXIT

mkdir -p "$OUT"/{logs,bench,meta,base,tip}
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== LAGUNA A/B base vs quality-safe tip $STAMP ==="
echo "OUT=$OUT"

export LD_LIBRARY_PATH="$BD:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_GRAPH=1
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1

{
  echo "stamp=$STAMP"
  echo "binary=$BD"
  md5sum "$BD"/libggml-sycl.so.0.17.0 2>/dev/null || true
  sha256sum "$MODEL" | awk '{print "model_sha256="$1}'
  echo "SRV_CTX=$SRV_CTX AGENT_CTX=$AGENT_CTX"
  echo "base_arm=all major custom fuses DISABLED"
  echo "tip_arm=quality-safe (only mm-add + dual-down + dual-multitoken disabled)"
} | tee "$OUT/meta/PINS.txt"

phase() { printf '\n[%s] PHASE %s\n' "$(date -Is)" "$1"; }

SRV_PID=
kill_port() {
  local p="$1" pids
  pids=$(ss -lptn "sport = :$p" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)
  for pid in $pids; do kill "$pid" 2>/dev/null || true; done
  sleep 1
  pids=$(ss -lptn "sport = :$p" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)
  for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
}
kill_srv() {
  if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill "$SRV_PID" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
  SRV_PID=
  kill_port "$PORT"
  sleep 2
}
trap 'kill_srv || true' EXIT INT TERM

ALL_KILLS=(
  GGML_SYCL_DISABLE_TOPK_MOE
  GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK
  GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK_NORM
  GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE
  GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD
  GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU
  GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN
  GGML_SYCL_DISABLE_MOE_DUAL_DOWN
  GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED
  GGML_SYCL_DISABLE_MOE_DOWN_INTEGRATED
  GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU
  GGML_SYCL_DISABLE_DENSE_DUAL_GEMM
  GGML_SYCL_DISABLE_RMS_NORM_FUSE
  GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE
  GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE
  GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE
  GGML_SYCL_DISABLE_ADD_ADD_FUSE
  GGML_SYCL_DISABLE_MOE_PACKED_REDUCE
)

apply_base() {
  for k in "${ALL_KILLS[@]}"; do export "$k=1"; done
  export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1
  export GGML_SYCL_DISABLE_GRAPH=1
}

apply_tip() {
  for k in "${ALL_KILLS[@]}"; do unset "$k" 2>/dev/null || true; done
  # quality-safe: keep only the three broken ones dead
  export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
  export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
  export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
  export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1
  export GGML_SYCL_DISABLE_GRAPH=1
}

start_server() {
  local tag="$1" ctx="${2:-$SRV_CTX}" npred="${3:-128}"
  kill_srv
  local jinja=(--no-jinja)
  [[ -f "$TEMPLATE" ]] && jinja=(--jinja --chat-template-file "$TEMPLATE")
  local alias="laguna-${tag}"
  echo "start arm=$tag ctx=$ctx alias=$alias"
  "$BD/llama-server" -m "$MODEL" -ngl 99 -t 16 \
    -c "$ctx" -np 1 -fa on -ctk f16 -ctv f16 \
    -b 2048 -ub 2048 -n "$npred" \
    --host 127.0.0.1 --port "$PORT" \
    "${jinja[@]}" --reasoning off --metrics -a "$alias" \
    >"$OUT/logs/server-${tag}-c${ctx}.log" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 180); do
    if curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
      echo "health ok ${tag} ~$((i*2))s"; return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "SERVER DIED $tag"; tail -50 "$OUT/logs/server-${tag}-c${ctx}.log"; return 1
    fi
    sleep 2
  done
  echo "health timeout $tag"; return 1
}

bench_formal() {
  local tag="$1"
  phase "formal $tag"
  local stamp_note="laguna-ab $tag"
  set +e
  (
    export LX_BIN="$BD" LX_LLAMA_BENCH="$BD/llama-bench" LD_LIBRARY_PATH="$BD:${LD_LIBRARY_PATH:-}"
    # re-export arm env into subshell
    env | rg '^GGML_SYCL_' || true
    "$ROOT/scripts/bench-serial.sh" --note "$stamp_note"
  )
  echo "formal_${tag}_exit=$?"
  set -e
  if [[ -f "$LX_RESULTS/LATEST_DIR.txt" ]]; then
    local ldir
    ldir=$(cat "$LX_RESULTS/LATEST_DIR.txt")
    mkdir -p "$OUT/$tag"
    cp -f "$ldir/metrics.json" "$OUT/$tag/formal-metrics.json" 2>/dev/null || true
    cp -f "$ldir/score.json" "$OUT/$tag/formal-score.json" 2>/dev/null || true
  fi
}

bench_ladder() {
  local tag="$1"
  phase "ladder $tag"
  set +e
  # Split pp/tg to reduce abort risk on large multi-test runs
  "$BD/llama-bench" -m "$MODEL" -ngl 99 -t 16 -ctk f16 -ctv f16 -fa on \
    -b 2048 -ub 2048 -p 512,2048,4096,8192 -n 0 -r 3 -o json \
    >"$OUT/$tag/ladder-pp.json" 2>"$OUT/$tag/ladder-pp.err"
  echo "ladder_pp_${tag}_exit=$?"
  "$BD/llama-bench" -m "$MODEL" -ngl 99 -t 16 -ctk f16 -ctv f16 -fa on \
    -b 2048 -ub 2048 -p 0 -n 128 -r 3 -o json \
    >"$OUT/$tag/ladder-tg.json" 2>"$OUT/$tag/ladder-tg.err"
  echo "ladder_tg_${tag}_exit=$?"
  python3 - <<'PY' "$OUT/$tag" || true
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
rows=[]
for name in ("ladder-pp.json","ladder-tg.json","ladder.json"):
  p=out/name
  if not p.exists() or p.stat().st_size<10: continue
  try:
    data=json.loads(p.read_text())
  except Exception as e:
    print(f"parse skip {name}: {e}"); continue
  chunk=data["results"] if isinstance(data,dict) and "results" in data else (data if isinstance(data,list) else [data])
  rows.extend([r for r in chunk if isinstance(r,dict)])
lines=["| n_prompt | n_gen | t/s |","|--------:|------:|----:|"]
for r in rows:
  lines.append(f"| {r.get('n_prompt','')} | {r.get('n_gen','')} | {r.get('avg_ts',0):.2f} |")
(out/"ladder.md").write_text("\n".join(lines)+"\n")
print((out/"ladder.md").read_text())
if rows:
  (out/"ladder.json").write_text(json.dumps({"results":rows}, indent=2)+"\n")
PY
  set -e
}

run_longctx() {
  local tag="$1"
  phase "longctx $tag"
  start_server "$tag" "$SRV_CTX" 64
  local alias="laguna-${tag}"
  mkdir -p "$OUT/$tag/longctx"
  set +e
  # longctx py resolves model from server
  python3 "$LONGCTX_PY" --base "$BASE" --out "$OUT/$tag/longctx" --needle-paragraphs 100
  echo "longctx_${tag}_exit=$?"
  set -e
  cp -f "$OUT/$tag/longctx/REPORT.md" "$OUT/$tag/longctx-REPORT.md" 2>/dev/null || true
}

run_single() {
  local tag="$1"
  phase "single-agent $tag"
  if ! curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
    start_server "$tag" "$SRV_CTX" 128
  fi
  local alias="laguna-${tag}"
  python3 - <<'PY' "$BASE" "$alias" "$OUT/$tag"
import json, time, statistics, urllib.request, sys
from pathlib import Path
base, alias, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
PROMPTS = [
  "Summarize in one sentence: latency vs throughput in LLM serving.",
  "Write a short Python function fib(n).",
  "List three risks of fusing GPU kernels without correctness gates.",
  "Explain MoE routing in two sentences.",
  "Log: ERROR migration check failed code=E_SCHEMA_MISSING — next debug command?",
]
def chat(msg, max_tokens=96):
  body=json.dumps({"model":alias,"messages":[{"role":"user","content":msg}],
    "max_tokens":max_tokens,"temperature":0,"seed":42,"stream":False,
    "chat_template_kwargs":{"enable_thinking":False}}).encode()
  req=urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions",data=body,
    headers={"Content-Type":"application/json"})
  t0=time.time()
  with urllib.request.urlopen(req,timeout=300) as r:
    d=json.loads(r.read().decode())
  tim=d.get("timings") or {}
  c=(d["choices"][0]["message"].get("content") or "")
  return {"wall_s":time.time()-t0,"tg":tim.get("predicted_per_second"),"pp":tim.get("prompt_per_second"),
          "ok":bool(c.strip()) and ".__" not in c[:20],"preview":c[:160].replace("\n"," ")}
rows=[]
for i,p in enumerate(PROMPTS):
  for rep in range(2):
    r=chat(p); r["tag"]=f"p{i}r{rep}"; rows.append(r); print(r, flush=True)
(out/"single.json").write_text(json.dumps(rows,indent=2)+"\n")
tgs=[float(r["tg"]) for r in rows if r.get("tg")]
summary={"tg_p50":statistics.median(tgs) if tgs else None,
         "tg_mean":statistics.mean(tgs) if tgs else None,
         "n_ok":sum(1 for r in rows if r.get("ok")),"n":len(rows)}
(out/"single_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("SINGLE", summary)
PY
}

run_heldout() {
  local tag="$1"
  phase "heldout $tag"
  start_server "$tag" "$SRV_CTX" 128
  local alias="laguna-${tag}"
  set +e
  python3 "$HELD_OUT/runner.py" --server-url "$BASE" --model "$alias" \
    --temperature 0 --seed 42 --max-turns 8 \
    --json-out "$OUT/$tag/heldout.json" \
    2>&1 | tee "$OUT/$tag/heldout.console.log"
  echo "heldout_${tag}_exit=$?"
  set -e
  if [[ -f "$OUT/$tag/heldout.json" ]]; then
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in ("score_pct","points","max_points","outcomes","n")})' "$OUT/$tag/heldout.json"
  fi
}

run_agent69() {
  local tag="$1"
  phase "agent69 $tag ctx=$AGENT_CTX"
  if [[ ! -x "$TOOL_BIN" ]]; then
    echo "SKIP agent69 no tool-eval"; return 0
  fi
  start_server "$tag" "$AGENT_CTX" 256 || { echo "server fail"; return 0; }
  local alias="laguna-${tag}"
  mkdir -p "$OUT/$tag/public69/runs"
  curl -s --max-time 60 "$BASE/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"ping","n_predict":4,"temperature":0}' >/dev/null || true
  set +e
  (
    cd "$TOOL_ROOT"
    timeout 9000 "$TOOL_BIN" --backend llamacpp \
      --base-url "$BASE" --model "$alias" \
      --temperature 0 --no-think --seed 42 --reference-date 2026-03-20 \
      --parallel 1 --timeout 180 --max-turns 8 \
      --output-dir "$OUT/$tag/public69/runs" \
      --json-file "$OUT/$tag/public69/result.json" \
      --no-live --redact-url
  ) 2>&1 | tee "$OUT/$tag/public69/console.log"
  echo "agent69_${tag}_exit=$?"
  set -e
  if [[ -f "$OUT/$tag/public69/result.json" ]]; then
    python3 - <<'PY' "$OUT/$tag/public69"
import json,sys
from collections import Counter
from pathlib import Path
out=Path(sys.argv[1])
d=json.loads((out/"result.json").read_text())
s=d.get("scores") or {}
sr=s.get("scenario_results") or []
c=Counter(x.get("status") for x in sr)
summary={
  "final_score": d.get("final_score"),
  "total_points": s.get("total_points"),
  "max_points": s.get("max_points"),
  "pass": c.get("pass",0),
  "partial": c.get("partial",0),
  "fail": c.get("fail",0),
  "n": len(sr),
  "median_turn_ms": s.get("median_turn_ms"),
  "ctx_exceed_errors": sum(1 for x in sr if "exceed" in str(x.get("summary") or "").lower() or "exceed" in str(x.get("error") or "").lower()),
}
# count context errors from console
console=(out/"console.log").read_text(errors="replace") if (out/"console.log").exists() else ""
summary["ctx_exceed_log_hits"]=console.count("exceed_context_size_error")
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
PY
  fi
}

run_arm() {
  local tag="$1"
  mkdir -p "$OUT/$tag"
  if [[ "$tag" == "base" ]]; then apply_base; else apply_tip; fi
  echo "=== ARM $tag env ==="
  env | rg '^GGML_SYCL_DISABLE' | sort | tee "$OUT/$tag/env.txt"
  bench_formal "$tag"
  bench_ladder "$tag"
  run_longctx "$tag"
  run_single "$tag"
  run_heldout "$tag"
  run_agent69 "$tag"
  kill_srv
}

# Order: base first, then tip
run_arm base
run_arm tip

# Assemble REPORT
phase "REPORT"
python3 - <<'PY' "$OUT"
import json
from pathlib import Path
from datetime import datetime, timezone
out=Path(__import__("sys").argv[1])

def loadj(*parts, default=None):
  p=out.joinpath(*parts)
  if p.exists():
    try: return json.loads(p.read_text())
    except Exception: return default
  return default

def formal(tag):
  s=loadj(tag,"formal-score.json") or {}
  m=loadj(tag,"formal-metrics.json") or {}
  return {
    "pp": s.get("prefill_tok_s") or m.get("pp512"),
    "tg": s.get("decode_tok_s") or m.get("tg128"),
    "pct": s.get("increase_pct"),
  }

def needles(tag):
  # parse REPORT
  rep=out/tag/"longctx-REPORT.md"
  if not rep.exists():
    rep=out/tag/"longctx"/"REPORT.md"
  text=rep.read_text() if rep.exists() else ""
  import re
  n=re.search(r"score:\s*\*\*(\d+)/(\d+)\*\*", text)
  # second score for dossier
  scores=re.findall(r"score:\s*\*\*(\d+)/(\d+)\*\*", text)
  needles=scores[0] if scores else (None,None)
  dossier=scores[1] if len(scores)>1 else (None,None)
  return {"needles": needles, "dossier": dossier, "text": text[:500]}

def held(tag):
  d=loadj(tag,"heldout.json") or {}
  return {"score_pct": d.get("score_pct", d.get("score")), "points": d.get("points", d.get("total_points")),
          "max": d.get("max_points"), "outcomes": d.get("outcomes")}

def a69(tag):
  return loadj(tag,"public69","summary.json") or {}

def single(tag):
  return loadj(tag,"single_summary.json") or {}

def ladder_pp(tag, n):
  p=out/tag/"ladder.json"
  if not p.exists(): return None
  data=json.loads(p.read_text())
  rows=data["results"] if isinstance(data,dict) and "results" in data else data
  for r in rows:
    if isinstance(r,dict) and r.get("n_prompt")==n and r.get("n_gen") in (0,"0",None):
      return r.get("avg_ts")
  return None

base_f, tip_f = formal("base"), formal("tip")
base_n, tip_n = needles("base"), needles("tip")
base_h, tip_h = held("base"), held("tip")
base_a, tip_a = a69("base"), a69("tip")
base_s, tip_s = single("base"), single("tip")

def delta(a,b):
  if a is None or b is None: return None
  try: return float(b)-float(a)
  except Exception: return None

lines=[
  "# Laguna A/B — base vs quality-safe tip",
  "",
  f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
  "",
  "Same binary + same Laguna Q4_K_M GGUF.",
  "",
  "| arm | meaning |",
  "|-----|---------|",
  "| **base** | all major custom SYCL fuses **OFF** |",
  "| **tip** | quality-safe tip (mm-add + dual-down + dual-multitoken OFF; rest ON) |",
  "",
  "## Scoreboard (tip − base)",
  "",
  "| Gate | base | tip | tip−base |",
  "|------|-----:|----:|---------:|",
  f"| formal pp512 | {base_f.get('pp')} | {tip_f.get('pp')} | {delta(base_f.get('pp'), tip_f.get('pp'))} |",
  f"| formal tg128 | {base_f.get('tg')} | {tip_f.get('tg')} | {delta(base_f.get('tg'), tip_f.get('tg'))} |",
  f"| vs pin % | {base_f.get('pct')} | {tip_f.get('pct')} | {delta(base_f.get('pct'), tip_f.get('pct'))} |",
  f"| ladder pp2048 | {ladder_pp('base',2048)} | {ladder_pp('tip',2048)} | {delta(ladder_pp('base',2048), ladder_pp('tip',2048))} |",
  f"| ladder pp8192 | {ladder_pp('base',8192)} | {ladder_pp('tip',8192)} | {delta(ladder_pp('base',8192), ladder_pp('tip',8192))} |",
  f"| single-agent tg p50 | {base_s.get('tg_p50')} | {tip_s.get('tg_p50')} | {delta(base_s.get('tg_p50'), tip_s.get('tg_p50'))} |",
  f"| single content ok | {base_s.get('n_ok')}/{base_s.get('n')} | {tip_s.get('n_ok')}/{tip_s.get('n')} |  |",
  f"| needles | {base_n['needles'][0]}/{base_n['needles'][1] if base_n['needles'][0] is not None else '?'} | {tip_n['needles'][0]}/{tip_n['needles'][1] if tip_n['needles'][0] is not None else '?'} |  |",
  f"| dossier | {base_n['dossier'][0]}/{base_n['dossier'][1] if base_n['dossier'][0] is not None else '?'} | {tip_n['dossier'][0]}/{tip_n['dossier'][1] if tip_n['dossier'][0] is not None else '?'} |  |",
  f"| held-out % | {base_h.get('score_pct')} | {tip_h.get('score_pct')} | {delta(base_h.get('score_pct'), tip_h.get('score_pct'))} |",
  f"| held-out points | {base_h.get('points')}/{base_h.get('max')} | {tip_h.get('points')}/{tip_h.get('max')} |  |",
  f"| agent69 score | {base_a.get('final_score')} | {tip_a.get('final_score')} | {delta(base_a.get('final_score'), tip_a.get('final_score'))} |",
  f"| agent69 pass/partial/fail | {base_a.get('pass')}/{base_a.get('partial')}/{base_a.get('fail')} | {tip_a.get('pass')}/{tip_a.get('partial')}/{tip_a.get('fail')} |  |",
  f"| agent69 ctx_exceed log hits | {base_a.get('ctx_exceed_log_hits')} | {tip_a.get('ctx_exceed_log_hits')} |  |",
  "",
  "Notes:",
  f"- Server ctx for long/heldout: see meta. Agent69 ctx may be limited by VRAM; ctx-exceed counts matter.",
  f"- Artifacts: `{out}`",
  "",
]
(out/"REPORT.md").write_text("\n".join(lines)+"\n")
print((out/"REPORT.md").read_text())
PY

echo "$OUT" >"$LX_RESULTS/LATEST_LAGUNA_AB_DIR.txt"
echo "DONE $OUT"
