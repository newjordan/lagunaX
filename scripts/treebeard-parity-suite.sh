#!/usr/bin/env bash
# Treebeard-parity proof on Laguna quality-safe tip (serial-first).
# Mirrors treebeard scorecard gates: formal, ladder, longctx, single-agent
# speed, held-out, Agent Bench 69. Multi-slot is optional appendix (np=4).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$LX_RESULTS/treebeard-parity-$STAMP}"
BD="${BD:-$LX_BIN}"
MODEL="${MODEL:-$LX_MODEL}"
PORT="${PORT:-18930}"
BASE="http://127.0.0.1:${PORT}"
ALIAS="${ALIAS:-laguna-quality-safe}"
TEMPLATE="${LX_TEMPLATE:-/home/frosty40/turbo/worktrees/treebeard-pr-private-latest/models/templates/poolside-Laguna-XS-2.1.jinja}"
TOOL_ROOT="${TOOL_ROOT:-/tmp/tool-eval-bench}"
TOOL_BIN="$TOOL_ROOT/.venv/bin/tool-eval-bench"
HELD_OUT="${HELD_OUT:-/home/frosty40/turbo/held-out-probe}"
LONGCTX_PY="${LONGCTX_PY:-/home/frosty40/turbo/treebeard-work/research/treebeard-pr-private/run-long-context-eval.py}"
# Laguna 19G Q4 on B70: 262k is not realistic; match single-agent AB scale
SRV_CTX="${SRV_CTX:-32768}"
SKIP_AGENT69="${SKIP_AGENT69:-0}"
SKIP_HELDOUT="${SKIP_HELDOUT:-0}"
SKIP_LONGCTX="${SKIP_LONGCTX:-0}"
SKIP_FORMAL="${SKIP_FORMAL:-0}"
SKIP_LADDER="${SKIP_LADDER:-0}"
SKIP_SINGLE="${SKIP_SINGLE:-0}"
SKIP_MULTISLOT="${SKIP_MULTISLOT:-0}"

mkdir -p "$OUT"/{logs,bench,longctx,agent,meta,single,multislot}
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== TREEBEARD-PARITY LAGUNA QUALITY-SAFE $STAMP ==="
echo "OUT=$OUT BD=$BD MODEL=$MODEL SRV_CTX=$SRV_CTX"
echo "kills: MUL_MAT_ADD=$GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE DUAL_DOWN=$GGML_SYCL_DISABLE_MOE_DUAL_DOWN DUAL_MT=$GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN"

export LD_LIBRARY_PATH="$BD:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0

{
  echo "stamp=$STAMP"
  echo "binary=$BD"
  ls -la "$BD/llama-bench" "$BD/llama-server" 2>&1 || true
  md5sum "$BD"/libggml-sycl.so.0.17.0 2>/dev/null || true
  sha256sum "$MODEL" | awk '{print "model_sha256="$1}'
  echo "GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=$GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE"
  echo "GGML_SYCL_DISABLE_MOE_DUAL_DOWN=$GGML_SYCL_DISABLE_MOE_DUAL_DOWN"
  echo "GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=$GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN"
  echo "GGML_SYCL_DISABLE_GRAPH=$GGML_SYCL_DISABLE_GRAPH"
  echo "SRV_CTX=$SRV_CTX"
  echo "template=$TEMPLATE"
  sycl-ls 2>&1 | head -5 || true
  if [[ -x "$TOOL_BIN" ]]; then "$TOOL_BIN" --version; git -C "$TOOL_ROOT" rev-parse HEAD; fi
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
cleanup() { kill_srv || true; }
trap cleanup EXIT INT TERM

start_server() {
  local ctx="${1:-$SRV_CTX}" np="${2:-1}" npred="${3:-256}"
  kill_srv
  local jinja_args=(--no-jinja)
  if [[ -f "$TEMPLATE" ]]; then
    jinja_args=(--jinja --chat-template-file "$TEMPLATE")
  fi
  echo "start server ctx=$ctx np=$np n_predict_default=$npred alias=$ALIAS"
  "$BD/llama-server" \
    -m "$MODEL" -ngl 99 -t 16 \
    -c "$ctx" -np "$np" -fa on -ctk f16 -ctv f16 \
    -b 2048 -ub 2048 -n "$npred" \
    --host 127.0.0.1 --port "$PORT" \
    "${jinja_args[@]}" \
    --reasoning off --metrics -a "$ALIAS" \
    >"$OUT/logs/server-c${ctx}-np${np}.log" 2>&1 &
  SRV_PID=$!
  echo "pid=$SRV_PID"
  for i in $(seq 1 200); do
    if curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
      echo "health ok after ~$((i*2))s"
      curl -s --max-time 5 "$BASE/health" | tee "$OUT/logs/health-c${ctx}-np${np}.json" >/dev/null || true
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "SERVER DIED"; tail -80 "$OUT/logs/server-c${ctx}-np${np}.log"; return 1
    fi
    sleep 2
  done
  echo "health timeout"; tail -80 "$OUT/logs/server-c${ctx}-np${np}.log"; return 1
}

# --- 1) Formal serial ---
if [[ "$SKIP_FORMAL" != "1" ]]; then
  phase "formal serial pp512/tg128"
  set +e
  (
    export LX_BIN="$BD" LX_LLAMA_BENCH="$BD/llama-bench" LD_LIBRARY_PATH="$BD:${LD_LIBRARY_PATH:-}"
    "$ROOT/scripts/bench-serial.sh" --note "treebeard-parity quality-safe formal"
  )
  echo "formal_exit=$?"
  set -e
  # copy latest into suite
  if [[ -f "$LX_RESULTS/LATEST_DIR.txt" ]]; then
    LDIR=$(cat "$LX_RESULTS/LATEST_DIR.txt")
    cp -f "$LDIR/metrics.json" "$OUT/bench/formal-metrics.json" 2>/dev/null || true
    cp -f "$LDIR/score.json" "$OUT/bench/formal-score.json" 2>/dev/null || true
  fi
fi

# --- 2) Prefill ladder ---
if [[ "$SKIP_LADDER" != "1" ]]; then
  phase "prefill ladder"
  set +e
  "$BD/llama-bench" -m "$MODEL" -ngl 99 -t 16 -ctk f16 -ctv f16 -fa on \
    -b 2048 -ub 2048 -p 512,2048,4096,8192,16384 -n 128 -r 3 -o json \
    >"$OUT/bench/ladder.json" 2>"$OUT/bench/ladder.err"
  echo "ladder_exit=$?"
  set -e
  python3 - <<'PY' "$OUT/bench/ladder.json" "$OUT/bench/ladder.md"
import json, sys
from pathlib import Path
src, md = Path(sys.argv[1]), Path(sys.argv[2])
if not src.exists() or src.stat().st_size < 10:
    md.write_text("# Ladder\n\nFAILED\n"); raise SystemExit(0)
data = json.loads(src.read_text())
rows = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [data])
lines = ["# Prefill / decode ladder", "", "| n_prompt | n_gen | t/s |", "|--------:|------:|----:|"]
for r in rows:
    if isinstance(r, dict):
        lines.append(f"| {r.get('n_prompt','')} | {r.get('n_gen','')} | {r.get('avg_ts',0):.2f} |")
md.write_text("\n".join(lines)+"\n"); print(md.read_text())
PY
fi

# --- 3) Long-context quality ---
if [[ "$SKIP_LONGCTX" != "1" ]]; then
  phase "long-context needles + dossier"
  start_server "$SRV_CTX" 1 64
  if [[ -f "$LONGCTX_PY" ]]; then
    paras=100
    set +e
    python3 "$LONGCTX_PY" --base "$BASE" --out "$OUT/longctx" --needle-paragraphs "$paras"
    echo "longctx_exit=$?"
    set -e
    cp -f "$OUT/longctx/REPORT.md" "$OUT/longctx-REPORT.md" 2>/dev/null || true
  fi
fi

# --- 4) Single-agent sequential speed (treebeard 1.2 style) ---
if [[ "$SKIP_SINGLE" != "1" ]]; then
  phase "single-agent sequential speed"
  if ! curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
    start_server "$SRV_CTX" 1 128
  fi
  python3 - <<'PY' "$BASE" "$ALIAS" "$OUT/single"
import json, time, statistics, urllib.request, sys
from pathlib import Path
base, alias, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Summarize in one sentence: the tradeoff between latency and throughput in LLM serving.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "List three risks of fusing GPU kernels without correctness gates.",
    "Explain MoE routing in two sentences for a systems engineer.",
    "Given log line 'ERROR migration check failed code=E_SCHEMA_MISSING', propose the next debug command.",
]

def chat(messages, max_tokens=96):
    data = json.dumps({
        "model": alias,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read().decode())
    wall = time.time() - t0
    tim = d.get("timings") or {}
    content = ""
    try:
        content = d["choices"][0]["message"].get("content") or ""
    except Exception:
        pass
    return {
        "wall_s": wall,
        "prompt_n": tim.get("prompt_n"),
        "predicted_n": tim.get("predicted_n"),
        "pp_tps": tim.get("prompt_per_second"),
        "tg_tps": tim.get("predicted_per_second"),
        "preview": content[:200].replace("\n", " "),
        "ok": bool(content.strip()) and not content.strip().startswith(".__"),
    }

rows = []
for i, p in enumerate(PROMPTS):
    for rep in range(2):
        r = chat([{"role": "user", "content": p}], 96)
        r["tag"] = f"p{i}_r{rep}"
        rows.append(r)
        print(json.dumps(r), flush=True)

(out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
tgs = [float(r["tg_tps"]) for r in rows if r.get("tg_tps")]
walls = [float(r["wall_s"]) for r in rows]
oks = sum(1 for r in rows if r.get("ok"))
summary = {
    "n": len(rows),
    "n_ok_content": oks,
    "tg_p50": statistics.median(tgs) if tgs else None,
    "tg_mean": statistics.mean(tgs) if tgs else None,
    "tg_min": min(tgs) if tgs else None,
    "tg_max": max(tgs) if tgs else None,
    "wall_p50": statistics.median(walls) if walls else None,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("SINGLE_AGENT", json.dumps(summary, indent=2))
PY
fi

# --- 5) Held-out ---
if [[ "$SKIP_HELDOUT" != "1" && -f "$HELD_OUT/runner.py" ]]; then
  phase "held-out ho-pack-v1.1"
  # hard cap generations so runaway cannot hang
  start_server "$SRV_CTX" 1 128
  set +e
  python3 "$HELD_OUT/runner.py" \
    --server-url "$BASE" \
    --model "$ALIAS" \
    --temperature 0 --seed 42 --max-turns 8 \
    --json-out "$OUT/agent/heldout.json" \
    2>&1 | tee "$OUT/agent/heldout.console.log"
  echo "heldout_exit=$?"
  set -e
  if [[ -f "$OUT/agent/heldout.json" ]]; then
    python3 - <<'PY' "$OUT/agent/heldout.json"
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({
  "score_pct": d.get("score_pct", d.get("score")),
  "points": d.get("points", d.get("total_points")),
  "max_points": d.get("max_points"),
  "outcomes": d.get("outcomes"),
  "n": d.get("n"),
}, indent=2))
PY
  fi
fi

# --- 6) Public Agent Bench 69 ---
if [[ "$SKIP_AGENT69" != "1" ]]; then
  phase "public Agent Bench 69"
  if [[ ! -x "$TOOL_BIN" ]]; then
    echo "SKIP agent69: missing $TOOL_BIN"
  else
    start_server "$SRV_CTX" 1 256
    # warmup
    curl -s --max-time 120 "$BASE/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"ping","n_predict":8,"temperature":0}' >"$OUT/agent/warmup.json" || true
    mkdir -p "$OUT/agent/public69/runs"
    set +e
    (
      cd "$TOOL_ROOT"
      timeout 9000 "$TOOL_BIN" --backend llamacpp \
        --base-url "$BASE" --model "$ALIAS" \
        --temperature 0 --no-think --seed 42 --reference-date 2026-03-20 \
        --parallel 1 --timeout 180 --max-turns 8 \
        --output-dir "$OUT/agent/public69/runs" \
        --json-file "$OUT/agent/public69/result.json" \
        --no-live --redact-url
    ) 2>&1 | tee "$OUT/agent/public69/console.log"
    echo "agent69_exit=$?"
    set -e
    if [[ -f "$OUT/agent/public69/result.json" ]]; then
      python3 - <<'PY' "$OUT/agent/public69"
import json, sys
from collections import Counter
from pathlib import Path
out = Path(sys.argv[1])
d = json.loads((out / "result.json").read_text())
s = d.get("scores") or {}
sr = s.get("scenario_results") or []
c = Counter(x.get("status") for x in sr)
summary = {
    "final_score": d.get("final_score"),
    "rating": d.get("rating"),
    "total_scenarios": d.get("total_scenarios"),
    "total_points": s.get("total_points"),
    "max_points": s.get("max_points"),
    "pass_count": c.get("pass", 0),
    "partial_count": c.get("partial", 0),
    "fail_count": c.get("fail", 0),
    "median_turn_ms": s.get("median_turn_ms"),
    "tool_eval_bench_version": d.get("tool_eval_bench_version"),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("non-pass:")
for x in sr:
    if x.get("status") in ("fail", "partial"):
        print(f"  {x.get('scenario_id')}: {x.get('status')} pts={x.get('points')} — {(x.get('summary') or '')[:100]}")
PY
    fi
  fi
fi

# --- 7) Multi-slot appendix (np=4, smaller than treebeard np12) ---
if [[ "$SKIP_MULTISLOT" != "1" ]]; then
  phase "multi-slot appendix np=4"
  set +e
  start_server 16384 4 64
  if [[ $? -eq 0 ]]; then
    python3 - <<'PY' "$BASE" "$ALIAS" "$OUT/multislot"
import json, time, urllib.request, concurrent.futures, statistics, sys
from pathlib import Path
base, alias, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

def one(i):
    body = json.dumps({
        "model": alias,
        "messages": [{"role": "user", "content": f"Agent {i}: reply with exactly three words."}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": 42 + i,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read().decode())
        tim = d.get("timings") or {}
        return {"i": i, "ok": True, "wall_s": time.time()-t0,
                "tg": tim.get("predicted_per_second"), "pp": tim.get("prompt_per_second"),
                "predicted_n": tim.get("predicted_n")}
    except Exception as e:
        return {"i": i, "ok": False, "error": str(e), "wall_s": time.time()-t0}

# two waves of 4 concurrent
rows = []
for wave in range(2):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, wave*4 + j) for j in range(4)]
        for f in concurrent.futures.as_completed(futs):
            rows.append(f.result())
            print(rows[-1], flush=True)
(out / "rows.json").write_text(json.dumps(rows, indent=2)+"\n")
tgs = [float(r["tg"]) for r in rows if r.get("ok") and r.get("tg")]
summary = {
    "n_ok": sum(1 for r in rows if r.get("ok")),
    "n": len(rows),
    "tg_p50_per_request": statistics.median(tgs) if tgs else None,
    "tg_mean_per_request": statistics.mean(tgs) if tgs else None,
    "note": "np=4 appendix only; not treebeard np12 fleet claim",
}
(out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print("MULTISLOT", json.dumps(summary, indent=2))
PY
  else
    echo "multislot server failed to start"
  fi
  set -e
fi

kill_srv

# --- Assemble SCORECARD ---
phase "assemble SCORECARD.md"
python3 - <<'PY' "$OUT"
import json, re
from pathlib import Path
from datetime import datetime, timezone
out = Path(__import__("sys").argv[1])
lines = [
    f"# Laguna quality-safe — Treebeard-parity scorecard",
    "",
    f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "Model: Laguna-XS-2.1 Q4_K_M · Binary: control tip · **quality-safe env kills**",
    "(MUL_MAT_ADD + MOE_DUAL_DOWN + MOE_DUAL_MULTITOKEN off).",
    "",
    "Not matched Qwen/Treebeard product A/B — same **gate instruments**, different model.",
    "",
    "## Pins",
    "",
    "```",
    (out/"meta"/"PINS.txt").read_text() if (out/"meta"/"PINS.txt").exists() else "",
    "```",
    "",
]

# formal
fs = out/"bench"/"formal-score.json"
if fs.exists():
    s = json.loads(fs.read_text())
    lines += [
        "## 1. Formal serial (pp512 / tg128)",
        "",
        f"| metric | value |",
        f"|--------|------:|",
        f"| pp512 | {s.get('prefill_tok_s',0):.1f} |",
        f"| tg128 | {s.get('decode_tok_s',0):.1f} |",
        f"| vs pin | **+{s.get('increase_pct',0):.2f}%** |",
        f"| decode× | {s.get('decode_speedup',0):.3f} |",
        f"| prefill× | {s.get('prefill_speedup',0):.3f} |",
        "",
    ]
elif (out/"bench"/"formal-metrics.json").exists():
    m = json.loads((out/"bench"/"formal-metrics.json").read_text())
    lines += ["## 1. Formal serial", "", f"pp={m.get('pp512')} tg={m.get('tg128')}", ""]

# ladder
if (out/"bench"/"ladder.md").exists():
    lines += ["## 2. Prefill ladder", "", (out/"bench"/"ladder.md").read_text(), ""]

# longctx
if (out/"longctx"/"REPORT.md").exists():
    lines += ["## 3. Long-context", "", (out/"longctx"/"REPORT.md").read_text(), ""]
elif (out/"longctx-REPORT.md").exists():
    lines += ["## 3. Long-context", "", (out/"longctx-REPORT.md").read_text(), ""]

# single agent
if (out/"single"/"summary.json").exists():
    s = json.loads((out/"single"/"summary.json").read_text())
    lines += [
        "## 4. Single-agent sequential speed",
        "",
        f"| metric | value |",
        f"|--------|------:|",
        f"| tg p50 | **{s.get('tg_p50')}** |",
        f"| tg mean | {s.get('tg_mean')} |",
        f"| wall p50 s | {s.get('wall_p50')} |",
        f"| content ok | {s.get('n_ok_content')}/{s.get('n')} |",
        "",
        "Treebeard reference (Qwen stock Q5): control 77.1 · package 88.9 tg p50 — **not comparable model**.",
        "",
    ]

# heldout
if (out/"agent"/"heldout.json").exists():
    d = json.loads((out/"agent"/"heldout.json").read_text())
    lines += [
        "## 5. Held-out ho-pack-v1.1",
        "",
        f"- score: **{d.get('score_pct', d.get('score'))}**",
        f"- points: {d.get('points', d.get('total_points'))}/{d.get('max_points')}",
        f"- outcomes: `{json.dumps(d.get('outcomes') or {})}`",
        "",
        "Treebeard reference: 91.3% (42/46) on Qwen — **not comparable model**.",
        "",
    ]

# agent69
if (out/"agent"/"public69"/"summary.json").exists():
    s = json.loads((out/"agent"/"public69"/"summary.json").read_text())
    lines += [
        "## 6. Public Agent Bench 69",
        "",
        f"| metric | value |",
        f"|--------|------:|",
        f"| final_score | **{s.get('final_score')}**/100 |",
        f"| points | {s.get('total_points')}/{s.get('max_points')} |",
        f"| pass / partial / fail | {s.get('pass_count')} / {s.get('partial_count')} / {s.get('fail_count')} |",
        f"| median_turn_ms | {s.get('median_turn_ms')} |",
        "",
        "Treebeard reference: 91/100 on Qwen — **not comparable model**.",
        "",
    ]

# multislot
if (out/"multislot"/"summary.json").exists():
    s = json.loads((out/"multislot"/"summary.json").read_text())
    lines += [
        "## 7. Multi-slot appendix (np=4)",
        "",
        f"- ok: {s.get('n_ok')}/{s.get('n')}",
        f"- tg p50 per request: {s.get('tg_p50_per_request')}",
        f"- note: {s.get('note')}",
        "",
        "Treebeard np12 fleet is a different claim (package multi-slot).",
        "",
    ]

lines += ["## Artifacts", "", f"`{out}`", ""]
(out/"SCORECARD.md").write_text("\n".join(lines)+"\n")
print((out/"SCORECARD.md").read_text())
print(f"\n=== SCORECARD → {out}/SCORECARD.md ===")
PY

echo "$OUT" >"$LX_RESULTS/LATEST_PARITY_DIR.txt"
echo "DONE $OUT"
