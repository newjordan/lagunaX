#!/usr/bin/env bash
# Run after GPU recover / reboot. Mount Doom resume.
# Note: env.sh already sources oneAPI setvars — do not re-source under set -e.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "1) device check"
# Avoid pipefail SIGPIPE from head closing early under set -o pipefail
sycl-ls >"$ROOT/results/.sycl-ls.txt" 2>&1 || true
head -20 "$ROOT/results/.sycl-ls.txt" || true
if ! grep -q 'Arc.*B70\|level_zero:gpu' "$ROOT/results/.sycl-ls.txt" 2>/dev/null; then
  log "WARN: B70 not visible in sycl-ls — aborting"
  exit 1
fi

log "2) short smoke (control)"
export LD_LIBRARY_PATH="$LX_BIN:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
timeout 180 "$LX_LLAMA_BENCH" -m "$LX_MODEL" -ngl 99 -t 16 -ub 2048 -b 2048 -fa on -r 1 -p 0 -n 32 -o md

PKG_BIN=/home/frosty40/turbo/worktrees/lx-serial-kernel-pkg/build-serial/bin
if [[ ! -x "$PKG_BIN/llama-bench" ]]; then
  log "3) finish package serial build"
  cmake --build /home/frosty40/turbo/worktrees/lx-serial-kernel-pkg/build-serial -j"$(nproc)" --target llama-bench
fi

log "4) binary A/B (control / package / wavefront / serial-pkg)"
OUT="$LX_RESULTS/post-reboot-ab-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0

# Write JSON blobs to files — embedding multi-MB json in python -c argv is fragile
ab() {
  local tag=$1 bin=$2
  if [[ ! -x "$bin/llama-bench" ]]; then
    log "  skip $tag (no llama-bench)"
    return 0
  fi
  export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"
  log "  arm $tag → $bin"
  if ! "$bin/llama-bench" -m "$LX_MODEL" -ngl 99 -t 16 -ub 4096 -b 8192 -ctk f16 -ctv f16 -fa on -r 5 -p 512 -n 0 -o json \
      >"$OUT/${tag}.pp.json" 2>"$OUT/${tag}.pp.err"; then
    log "  FAIL $tag pp512 (see ${tag}.pp.err)"
    return 0
  fi
  if ! "$bin/llama-bench" -m "$LX_MODEL" -ngl 99 -t 16 -ub 4096 -b 8192 -ctk f16 -ctv f16 -fa on -r 5 -p 0 -n 128 -o json \
      >"$OUT/${tag}.tg.json" 2>"$OUT/${tag}.tg.err"; then
    log "  FAIL $tag tg128 (see ${tag}.tg.err)"
    return 0
  fi
  python3 - "$OUT" "$tag" <<'PY'
import json, sys
from pathlib import Path
out, tag = Path(sys.argv[1]), sys.argv[2]

def avg(path: Path, kind: str) -> float:
    data = json.loads(path.read_text())
    rows = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [data])
    for r in rows:
        a = r.get("avg_ts")
        if a is None:
            continue
        ng, np_, t = r.get("n_gen"), r.get("n_prompt"), str(r.get("test") or "")
        if kind == "pp" and (t.startswith("pp") or ng in (0, "0", None)):
            return float(a)
        if kind == "tg" and (t.startswith("tg") or np_ in (0, "0", None)):
            return float(a)
    return float(rows[0]["avg_ts"])

pp = avg(out / f"{tag}.pp.json", "pp")
tg = avg(out / f"{tag}.tg.json", "tg")
(out / f"{tag}.json").write_text(json.dumps({"tag": tag, "pp512": pp, "tg128": tg}, indent=2) + "\n")
print(f"  {tag} pp={pp:.2f} tg={tg:.2f}")
PY
}

ab ctrl /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
ab pkg /home/frosty40/turbo/worktrees/treebeard-pr-private-latest/build-positive-package/bin
ab wf /home/frosty40/turbo/treebeard-work/build-treebeard-single-wavefront/bin
ab serial_pkg "$PKG_BIN"

python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in out.glob("*.json"):
    if p.name in ("SUMMARY.json",) or p.name.endswith(".pp.json") or p.name.endswith(".tg.json"):
        continue
    try:
        rows.append(json.loads(p.read_text()))
    except Exception as e:
        print(f"skip {p.name}: {e}")
rows = sorted(rows, key=lambda r: r.get("tg128", 0), reverse=True)
(out / "SUMMARY.json").write_text(json.dumps(rows, indent=2) + "\n")
print("=== POST-REBOOT A/B ===")
for r in rows:
    print(f"{r['tag']:12} pp={r['pp512']:8.2f} tg={r['tg128']:7.2f}")
if not rows:
    print("(no successful arms)")
    sys.exit(1)
PY

log "5) relaunch quest daemon"
bash "$ROOT/scripts/quest-launch.sh"
log "DONE → $OUT"
