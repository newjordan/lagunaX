#!/usr/bin/env bash
# Laguna tip proof suite: formal speed + prefill ladder + ppl + long-ctx + agent.
# Serial track only. Writes results/proof-<stamp>/PROOF.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$LX_RESULTS/proof-$STAMP}"
BD="${BD:-$LX_BIN}"
MODEL="${MODEL:-$LX_MODEL}"
PORT="${PORT:-18911}"
BASE="http://127.0.0.1:${PORT}"
ALIAS="${ALIAS:-laguna-tip-proof}"
WIKI="${WIKI:-/home/frosty40/data/wikitext-2-raw/wiki.test.raw}"
TOOL_ROOT="${TOOL_ROOT:-/tmp/tool-eval-bench}"
TOOL_BIN="$TOOL_ROOT/.venv/bin/tool-eval-bench"
HELD_OUT="${HELD_OUT:-/home/frosty40/turbo/held-out-probe}"
TEMPLATE="${LX_TEMPLATE:-/home/frosty40/turbo/worktrees/treebeard-pr-private-latest/models/templates/poolside-Laguna-XS-2.1.jinja}"
LONGCTX_PY="${LONGCTX_PY:-/home/frosty40/turbo/treebeard-work/research/treebeard-pr-private/run-long-context-eval.py}"
# Laguna 19G Q4 on B70: keep server ctx practical (f16 KV)
SRV_CTX="${SRV_CTX:-16384}"
PPL_CHUNKS="${PPL_CHUNKS:-32}"
PPL_CTX="${PPL_CTX:-2048}"
SKIP_AGENT69="${SKIP_AGENT69:-0}"
SKIP_HELDOUT="${SKIP_HELDOUT:-0}"
SKIP_PPL="${SKIP_PPL:-0}"
SKIP_LADDER="${SKIP_LADDER:-0}"
SKIP_LONGCTX="${SKIP_LONGCTX:-0}"
SKIP_AGENT_TPUT="${SKIP_AGENT_TPUT:-0}"

# Whole suite holds the card — never interleave with other B70 jobs.
lx_gpu_lock_enter "proof-suite" || exit $?
trap 'lx_gpu_lock_leave' EXIT

mkdir -p "$OUT"/{logs,bench,ppl,longctx,agent,meta}
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== LAGUNA TIP PROOF SUITE $STAMP ==="
echo "OUT=$OUT BD=$BD MODEL=$MODEL SRV_CTX=$SRV_CTX"

export LD_LIBRARY_PATH="$BD:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_GRAPH=1
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1
# Quality-safe tip (see notes/SHIP_20260730_quality_safe_tip.md)
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE="${GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE:-1}"
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN="${GGML_SYCL_DISABLE_MOE_DUAL_DOWN:-1}"
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN="${GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN:-1}"
unset GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK_NORM || true
unset GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE || true

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

phase() { printf '\n[%s] PHASE %s\n' "$(date -Is)" "$1"; }

{
  echo "stamp=$STAMP"
  echo "binary=$BD/llama-bench"
  ls -la "$BD/llama-bench" "$BD/llama-server" "$BD/llama-perplexity" 2>&1 || true
  md5sum "$BD/libggml-sycl.so.0.17.0" 2>/dev/null || md5sum "$BD"/libggml-sycl.so* 2>/dev/null | head -3
  echo "model=$MODEL"
  sha256sum "$MODEL" | awk '{print "model_sha256="$1}'
  echo "wiki=$WIKI"
  echo "srv_ctx=$SRV_CTX"
  echo "template=$TEMPLATE"
  sycl-ls 2>&1 | head -5 || true
} | tee "$OUT/meta/PINS.txt"

# --- 1) Prefill ladder + formal-ish tg ---
if [[ "$SKIP_LADDER" != "1" ]]; then
  phase "prefill ladder (llama-bench)"
  set +e
  "$BD/llama-bench" \
    -m "$MODEL" -ngl 99 -t 16 \
    -ctk f16 -ctv f16 -fa on \
    -b 2048 -ub 2048 \
    -p 512,2048,4096,8192 \
    -n 128 \
    -r 3 \
    -o json \
    >"$OUT/bench/ladder.json" 2>"$OUT/bench/ladder.err"
  ec=$?
  set -e
  echo "ladder_exit=$ec"
  python3 - <<'PY' "$OUT/bench/ladder.json" "$OUT/bench/ladder.md"
import json, sys
from pathlib import Path
src, md = Path(sys.argv[1]), Path(sys.argv[2])
if not src.exists() or src.stat().st_size < 10:
    md.write_text("# Prefill ladder\n\nFAILED (no json)\n")
    raise SystemExit(0)
data = json.loads(src.read_text())
rows = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [data])
lines = ["# Prefill / decode ladder", "", "| test | n_prompt | n_gen | t/s |", "|------|--------:|------:|----:|"]
for r in rows:
    if not isinstance(r, dict):
        continue
    lines.append(f"| {r.get('test','')} | {r.get('n_prompt','')} | {r.get('n_gen','')} | {r.get('avg_ts',0):.2f} |")
md.write_text("\n".join(lines) + "\n")
print(md.read_text())
PY
fi

# --- 2) Perplexity ---
if [[ "$SKIP_PPL" != "1" ]]; then
  phase "perplexity wikitext-2 (chunks=$PPL_CHUNKS ctx=$PPL_CTX)"
  test -f "$WIKI"
  set +e
  "$BD/llama-perplexity" \
    -m "$MODEL" -f "$WIKI" \
    -ngl 99 -t 16 \
    -c "$PPL_CTX" -b 2048 -ub 512 \
    -ctk f16 -ctv f16 -fa on \
    --chunks "$PPL_CHUNKS" \
    2>"$OUT/ppl/ppl.err" | tee "$OUT/ppl/ppl.log"
  ec=$?
  set -e
  echo "ppl_exit=$ec"
  # extract final PPL line
  rg -n 'perplexity|PPL|Final' "$OUT/ppl/ppl.log" "$OUT/ppl/ppl.err" 2>/dev/null | tail -20 | tee "$OUT/ppl/summary.txt" || true
fi

# --- server for longctx + agent ---
start_server() {
  local ctx="$1"
  kill_srv
  local jinja_args=(--no-jinja)
  if [[ -f "$TEMPLATE" ]]; then
    jinja_args=(--jinja --chat-template-file "$TEMPLATE")
  fi
  echo "start server ctx=$ctx alias=$ALIAS"
  "$BD/llama-server" \
    -m "$MODEL" -ngl 99 -t 16 \
    -c "$ctx" -np 1 -fa on -ctk f16 -ctv f16 \
    -b 2048 -ub 2048 \
    --host 127.0.0.1 --port "$PORT" \
    "${jinja_args[@]}" \
    --metrics -a "$ALIAS" \
    >"$OUT/logs/server.log" 2>&1 &
  SRV_PID=$!
  echo "pid=$SRV_PID"
  for i in $(seq 1 180); do
    if curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
      echo "health ok after ~$((i*2))s"
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "SERVER DIED"; tail -80 "$OUT/logs/server.log"; return 1
    fi
    sleep 2
  done
  echo "health timeout"; tail -80 "$OUT/logs/server.log"; return 1
}

# --- 3) Long context quality ---
if [[ "$SKIP_LONGCTX" != "1" ]]; then
  phase "long-context needles + dossier"
  start_server "$SRV_CTX"
  if [[ -f "$LONGCTX_PY" ]]; then
    # smaller haystack than package defaults if ctx is 16k
    paras=80
    if [[ "$SRV_CTX" -ge 32768 ]]; then paras=150; fi
    set +e
    python3 "$LONGCTX_PY" \
      --base "$BASE" \
      --out "$OUT/longctx" \
      --needle-paragraphs "$paras"
    echo "longctx_exit=$?"
    set -e
    cp -f "$OUT/longctx/REPORT.md" "$OUT/longctx-REPORT.md" 2>/dev/null || true
  else
    echo "missing LONGCTX_PY=$LONGCTX_PY"
  fi
fi

# --- 4) Agent throughput (tool turns + depth) ---
if [[ "$SKIP_AGENT_TPUT" != "1" ]]; then
  phase "agent throughput (chat completions)"
  # ensure server up
  if ! curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
    start_server "$SRV_CTX"
  fi
  python3 - <<'PY' "$BASE" "$ALIAS" "$OUT/agent"
import json, time, urllib.request, statistics, sys
from pathlib import Path

base, alias, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

def post(path, body, timeout=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

SYS = ("You are AgentWorld, an agent operating a computer and web environment. "
       "Observe tool results, reason briefly, then either call a tool or answer. Be concise.")

TOOL_TURNS = [
    "I ran `ls -la /var/log && grep -c ERROR /var/log/app.log`. Output:\n```\n-rw-r--r-- 1 root root 18234 app.log\n47\n```\nSummarize and propose next command.",
    'The get_weather tool returned: {"location":"Austin, TX","temp_f":98,"humidity":0.41}. Write the user-facing reply.',
    "After clicking Checkout the DOM shows fields [email, card_number, expiry, cvc] and disabled Pay. Describe UI state and next action.",
]

FILLER = ("OBSERVE step={i}: process pid={i} rss={i}MB state=running cmd=worker --shard {i} "
          "--queue tasks-{i} latency_ms={i} ok=true\n")

def one(messages, max_tokens, tag):
    t0 = time.time()
    try:
        d = post("/v1/chat/completions", {
            "model": alias,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "stream": False,
            "cache_prompt": False,
        })
        wall = time.time() - t0
        # llama.cpp may put timings at top-level or nested
        tim = d.get("timings") or {}
        usage = d.get("usage") or {}
        content = ""
        try:
            content = d["choices"][0]["message"]["content"] or ""
        except Exception:
            pass
        row = {
            "tag": tag,
            "ok": True,
            "wall_s": wall,
            "prompt_n": tim.get("prompt_n") or usage.get("prompt_tokens"),
            "predicted_n": tim.get("predicted_n") or usage.get("completion_tokens"),
            "pp_tps": tim.get("prompt_per_second"),
            "tg_tps": tim.get("predicted_per_second"),
            "preview": content[:160].replace("\n", " "),
        }
    except Exception as e:
        row = {"tag": tag, "ok": False, "error": str(e), "wall_s": time.time() - t0}
    print(json.dumps(row), flush=True)
    return row

rows = []
# short agent tool turns (decode-heavy)
for i, u in enumerate(TOOL_TURNS):
    for rep in range(2):
        rows.append(one([
            {"role": "system", "content": SYS},
            {"role": "user", "content": u},
        ], 96, f"tool{i}_r{rep}"))

# depth-ish prefill agent turns
for depth in (2048, 8192):
    n = max(1, depth // 27)
    obs = "".join(FILLER.format(i=i) for i in range(n))
    rows.append(one([
        {"role": "system", "content": SYS},
        {"role": "user", "content": (
            "Here is the environment log. Report the worker with highest latency_ms and one action.\n\n" + obs
        )},
    ], 64, f"depth_{depth}"))

(out / "throughput.json").write_text(json.dumps(rows, indent=2) + "\n")

ok = [r for r in rows if r.get("ok") and r.get("tg_tps")]
tg = [float(r["tg_tps"]) for r in ok if r["tag"].startswith("tool")]
pp = [float(r["pp_tps"]) for r in ok if r.get("pp_tps") and r["tag"].startswith("depth")]
summary = {
    "n_ok": sum(1 for r in rows if r.get("ok")),
    "n_total": len(rows),
    "tool_tg_tps": {
        "n": len(tg),
        "mean": statistics.mean(tg) if tg else None,
        "p50": statistics.median(tg) if tg else None,
        "min": min(tg) if tg else None,
        "max": max(tg) if tg else None,
    },
    "depth_pp_tps": {
        "n": len(pp),
        "mean": statistics.mean(pp) if pp else None,
        "values": [(r["tag"], r.get("prompt_n"), r.get("pp_tps"), r.get("tg_tps")) for r in ok if r["tag"].startswith("depth")],
    },
}
(out / "throughput_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("SUMMARY", json.dumps(summary, indent=2))
PY
fi

# --- 5) Held-out pack ---
if [[ "$SKIP_HELDOUT" != "1" && -f "$HELD_OUT/runner.py" ]]; then
  phase "held-out ho-pack"
  if ! curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
    start_server "$SRV_CTX"
  fi
  set +e
  python3 "$HELD_OUT/runner.py" \
    --server-url "$BASE" \
    --model "$ALIAS" \
    --temperature 0 --seed 42 --max-turns 8 \
    --json-out "$OUT/agent/heldout.json" \
    2>&1 | tee "$OUT/agent/heldout.console.log"
  echo "heldout_exit=$?"
  set -e
fi

# --- 6) Public Agent Bench 69 ---
if [[ "$SKIP_AGENT69" != "1" ]]; then
  phase "public Agent Bench 69"
  if [[ ! -x "$TOOL_BIN" ]]; then
    echo "SKIP agent69: missing $TOOL_BIN"
  else
    if ! curl -s --max-time 2 "$BASE/health" 2>/dev/null | grep -q ok; then
      # agent tools often need larger ctx; try SRV_CTX first
      start_server "$SRV_CTX"
    fi
    # warmup completion so first tool-eval request is not cold-load only
    curl -s --max-time 120 "$BASE/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"ping","n_predict":8,"temperature":0}' >"$OUT/agent/warmup.json" || true
    mkdir -p "$OUT/agent/public69/runs"
    set +e
    (
      cd "$TOOL_ROOT"
      timeout 7200 "$TOOL_BIN" --backend llamacpp \
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
        print(f"  {x.get('scenario_id')}: {x.get('status')} pts={x.get('points')} — {(x.get('summary') or '')[:120]}")
PY
    fi
  fi
fi

kill_srv

# --- Assemble PROOF.md ---
phase "assemble PROOF.md"
python3 - <<'PY' "$OUT" "$ROOT"
import json, re, sys
from pathlib import Path
from datetime import datetime, timezone

out = Path(sys.argv[1])
root = Path(sys.argv[2])
lines = [
    f"# Laguna tip proof suite — {out.name}",
    "",
    f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "Serial track only (one stream). Model: Laguna-XS-2.1 Q4_K_M. Device: Arc Pro B70 SYCL.",
    "",
    "## 0. Pins",
    "",
    "```",
    (out / "meta" / "PINS.txt").read_text() if (out / "meta" / "PINS.txt").exists() else "(missing)",
    "```",
    "",
]

# formal latest
latest = root / "results" / "LATEST_SCORE.json"
if latest.exists():
    sc = json.loads(latest.read_text())
    lines += [
        "## 1. Formal serial (pp512 / tg128)",
        "",
        f"| metric | value |",
        f"|--------|------:|",
        f"| pp512 | {sc.get('prefill_tok_s',0):.1f} |",
        f"| tg128 | {sc.get('decode_tok_s',0):.1f} |",
        f"| score vs pin | **+{sc.get('increase_pct',0):.2f}%** |",
        f"| decode speedup | {sc.get('decode_speedup',0):.3f}x |",
        f"| prefill speedup | {sc.get('prefill_speedup',0):.3f}x |",
        f"| floors_ok | {sc.get('floors_ok')} |",
        f"| candidate | `{sc.get('candidate_path','')}` |",
        "",
    ]

# ladder
ladder_md = out / "bench" / "ladder.md"
if ladder_md.exists():
    lines += ["## 2. Prefill / decode ladder", "", ladder_md.read_text(), ""]

# ppl
ppl_sum = out / "ppl" / "summary.txt"
ppl_log = out / "ppl" / "ppl.log"
ppl_err = out / "ppl" / "ppl.err"
ppl_text = ""
for p in (ppl_sum, ppl_log, ppl_err):
    if p.exists():
        ppl_text += p.read_text(errors="replace") + "\n"
m = re.findall(r"[Pp]erplexity[^\n]*", ppl_text)
# also Final estimate
m2 = re.findall(r"Final estimate[^\n]*", ppl_text)
lines += ["## 3. Perplexity (wikitext-2)", ""]
if m or m2:
    lines.append("```")
    for x in (m + m2)[-12:]:
        lines.append(x)
    lines.append("```")
else:
    lines.append("_No PPL line parsed — see `ppl/` logs._")
lines.append("")

# longctx
lc_report = out / "longctx" / "REPORT.md"
if lc_report.exists():
    lines += ["## 4. Long-context quality", "", lc_report.read_text(), ""]
else:
    # try raw jsons
    lines += ["## 4. Long-context quality", "", "_See `longctx/` artifacts._", ""]

# agent tput
ts = out / "agent" / "throughput_summary.json"
if ts.exists():
    s = json.loads(ts.read_text())
    tt = s.get("tool_tg_tps") or {}
    lines += [
        "## 5. Agent throughput (np=1 chat)",
        "",
        f"| metric | value |",
        f"|--------|------:|",
        f"| tool-turn tg mean | {tt.get('mean')} |",
        f"| tool-turn tg p50 | {tt.get('p50')} |",
        f"| tool-turn tg min/max | {tt.get('min')} / {tt.get('max')} |",
        f"| requests ok | {s.get('n_ok')}/{s.get('n_total')} |",
        "",
        "Depth prefill samples:",
        "```json",
        json.dumps(s.get("depth_pp_tps"), indent=2),
        "```",
        "",
    ]

# heldout
ho = out / "agent" / "heldout.json"
if ho.exists():
    d = json.loads(ho.read_text())
    lines += [
        "## 6. Held-out pack",
        "",
        f"- score_pct: **{d.get('score_pct', d.get('score'))}**",
        f"- points: {d.get('points', d.get('total_points'))}/{d.get('max_points')}",
        f"- outcomes: `{json.dumps(d.get('outcomes') or {})}`",
        "",
    ]

# agent69
a69 = out / "agent" / "public69" / "summary.json"
if a69.exists():
    s = json.loads(a69.read_text())
    lines += [
        "## 7. Public Agent Bench 69",
        "",
        f"| metric | value |",
        f"|--------|------:|",
        f"| final_score | **{s.get('final_score')}**/100 |",
        f"| points | {s.get('total_points')}/{s.get('max_points')} |",
        f"| pass / partial / fail | {s.get('pass_count')} / {s.get('partial_count')} / {s.get('fail_count')} |",
        f"| median_turn_ms | {s.get('median_turn_ms')} |",
        f"| tool-eval | {s.get('tool_eval_bench_version')} |",
        "",
        "> Note: Agent Bench is a **tool-calling quality** gate. Laguna tip stack is a **kernel speed**",
        "> campaign on Q4 serial; do not equate score with Qwen package scorecards.",
        "",
    ]
elif (out / "agent" / "public69" / "console.log").exists():
    lines += ["## 7. Public Agent Bench 69", "", "_Ran but no summary — see `agent/public69/console.log`._", ""]

lines += [
    "## Artifacts",
    "",
    f"Directory: `{out}`",
    "",
    "- `bench/` prefill ladder",
    "- `ppl/` perplexity",
    "- `longctx/` needles + dossier",
    "- `agent/` throughput + held-out + public69",
    "- `logs/server.log`",
    "",
]
(out / "PROOF.md").write_text("\n".join(lines) + "\n")
print((out / "PROOF.md").read_text())
print(f"\n=== PROOF WRITTEN → {out}/PROOF.md ===")
PY

echo "$OUT" >"$LX_RESULTS/LATEST_PROOF_DIR.txt"
echo "DONE $OUT"
