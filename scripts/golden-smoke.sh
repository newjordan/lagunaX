#!/usr/bin/env bash
# Greedy correctness smoke for serial track (via llama-server; control bin has no llama-cli).
# First run with --capture writes golden; subsequent runs must match.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

N_TOKENS="${N_TOKENS:-64}"
PROMPT="${PROMPT:-Write a Python function that returns the nth Fibonacci number.}"
GOLDEN_PATH="${LX_GOLDEN}"
PORT="${LX_GOLDEN_PORT:-18901}"
CAPTURE=0
[[ "${1:-}" == "--capture" ]] && CAPTURE=1

if [[ ! -x "$LX_LLAMA_SERVER" ]]; then
  echo "missing llama-server: $LX_LLAMA_SERVER" >&2
  exit 1
fi

# Exclusive B70 — do not golden while bench/ppl owns the card.
lx_gpu_lock_enter "golden-smoke" || exit $?

TMP="$(mktemp -d)"
trap 'kill $(cat "$TMP/server.pid" 2>/dev/null) 2>/dev/null || true; rm -rf "$TMP"; lx_gpu_lock_leave' EXIT

echo "starting server on :$PORT ..."
# Laguna GGUF embeds a jinja template this binary can't parse; override with known-good file.
TEMPLATE="${LX_TEMPLATE:-/home/frosty40/turbo/worktrees/treebeard-pr-private-latest/models/templates/poolside-Laguna-XS-2.1.jinja}"
JINJA_ARGS=(--no-jinja)
if [[ -f "$TEMPLATE" ]]; then
  JINJA_ARGS=(--jinja --chat-template-file "$TEMPLATE")
fi

"$LX_LLAMA_SERVER" \
  -m "$LX_MODEL" \
  -ngl "$NGL" \
  -t "$THREADS" \
  -c 4096 \
  -ub "$UBATCH" \
  -b "$BBATCH" \
  -ctk "$CTK" \
  -ctv "$CTV" \
  --host 127.0.0.1 \
  --port "$PORT" \
  "${JINJA_ARGS[@]}" \
  --log-disable \
  >"$TMP/server.log" 2>&1 &
echo $! >"$TMP/server.pid"

# wait for health (model load ~20s+)
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$(cat "$TMP/server.pid")" 2>/dev/null; then
    echo "server died during load" >&2
    tail -40 "$TMP/server.log" >&2 || true
    exit 1
  fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "server health timeout" >&2
  tail -40 "$TMP/server.log" >&2 || true
  exit 1
fi

# greedy completion (native completion API; also try OpenAI-style)
python3 - "$PORT" "$PROMPT" "$N_TOKENS" "$TMP/resp.json" <<'PY'
import json, sys, urllib.request
port, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = sys.argv[4]
body = json.dumps({
    "prompt": prompt,
    "n_predict": n,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "seed": 42,
    "cache_prompt": False,
}).encode()
# prefer native /completion
for path in ("/completion", "/v1/completions"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            open(out, "wb").write(r.read())
        print(f"used {path}", file=sys.stderr)
        break
    except Exception as e:
        last = e
else:
    raise SystemExit(f"completion failed: {last}")
PY

python3 - "$TMP/resp.json" "$TMP/tokens.txt" <<'PY'
import json, pathlib, sys
resp = json.loads(pathlib.Path(sys.argv[1]).read_text())
# openai-ish or llama.cpp native
text = None
if "choices" in resp:
    ch = resp["choices"][0]
    text = ch.get("text") or (ch.get("message") or {}).get("content")
if text is None and "content" in resp:
    text = resp["content"]
if text is None:
    raise SystemExit(f"unparsed response keys: {list(resp.keys())}")
lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
key = "\n".join(lines[:40])
pathlib.Path(sys.argv[2]).write_text(key + "\n")
print(key[:500])
PY

KEY="$(cat "$TMP/tokens.txt")"

if [[ "$CAPTURE" -eq 1 || ! -f "$GOLDEN_PATH" ]]; then
  python3 - "$GOLDEN_PATH" "$KEY" "$PROMPT" "$N_TOKENS" <<'PY'
import json, sys, pathlib, datetime
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "captured_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "prompt": sys.argv[3],
    "n_tokens": int(sys.argv[4]),
    "temp": 0,
    "seed": 42,
    "generation_key": sys.argv[2],
    "note": "Local greedy smoke via llama-server — not mlx.fast hidden goldens.",
}
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"captured golden → {path}")
PY
  exit 0
fi

python3 - "$GOLDEN_PATH" "$KEY" <<'PY'
import json, sys, pathlib
golden = json.loads(pathlib.Path(sys.argv[1]).read_text())
got = sys.argv[2].strip()
want = golden["generation_key"].strip()
if got == want:
    print("GOLDEN OK (greedy smoke match)")
    raise SystemExit(0)
print("GOLDEN MISMATCH")
print("--- want (first 400) ---")
print(want[:400])
print("--- got (first 400) ---")
print(got[:400])
raise SystemExit(1)
PY
