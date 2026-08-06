#!/usr/bin/env bash
# ab-aggregate.sh — fold all A/B interleave windows (results/ab-*/receipt.json)
# into one per-variant verdict: mean pp512/tg128 delta vs same-window champion,
# window count, and a recommendation. Only variants with >=2 windows count.
# Usage: bash scripts/ab-aggregate.sh [results-root]
set -uo pipefail
ROOT="${1:-results}"
cd "$(dirname "$0")/.."

python3 - "$ROOT" <<'PY'
import json, glob, os, sys

root = sys.argv[1]
agg = {}  # variant -> {pp:[deltas], tg:[deltas], n, wins}
for f in sorted(glob.glob(os.path.join(root, "ab-*/receipt.json"))):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    v = r.get("candidate_env")
    if not v:
        continue
    # per-run deltas are already in the receipt
    ppd = r.get("pp_delta_pct")
    tgd = r.get("tg_delta_pct")
    if ppd is None or tgd is None:
        continue
    d = agg.setdefault(v, {"pp": [], "tg": [], "n": 0, "wins": []})
    d["pp"].append(ppd)
    d["tg"].append(tgd)
    d["n"] += 1
    d["wins"].append(os.path.basename(os.path.dirname(f)))

print(f"{'variant':48s} {'n':>3s} {'ppΔ%':>8s} {'tgΔ%':>8s}  verdict")
for v, d in sorted(agg.items(), key=lambda kv: -((sum(kv[1]["tg"])/len(kv[1]["tg"])) if kv[1]["tg"] else 0)):
    ppm = sum(d["pp"])/len(d["pp"]) if d["pp"] else float("nan")
    tgm = sum(d["tg"])/len(d["tg"]) if d["tg"] else float("nan")
    ver = "PROMOTE" if tgm >= 1.0 and ppm >= -0.5 and d["n"] >= 2 else ("PROBE" if d["n"] < 2 else "dead")
    print(f"{v:48s} {d['n']:3d} {ppm:8.2f} {tgm:8.2f}  {ver}")
PY
