#!/usr/bin/env bash
# Measure the decode ceiling from reducing Laguna's output-vocabulary projection.
# Nonzero row limits are research-only and MUST NOT be used for quality acceptance.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMITS="${LM_HEAD_LIMITS:-0 65536 32768 16384 8192 4096}"
OUT_ROOT="${LX_RESULTS:-$ROOT/results}/lm-head-sweep-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_ROOT"

for limit in $LIMITS; do
    arm="full"
    unset GGML_SYCL_LM_HEAD_ROW_LIMIT
    if (( limit > 0 )); then
        arm="rows-$limit"
        export GGML_SYCL_LM_HEAD_ROW_LIMIT="$limit"
    fi

    out="$OUT_ROOT/$arm"
    mkdir -p "$out"
    echo "== $arm =="
    LX_RESULTS="$out" "$ROOT/scripts/bench-serial.sh" candidate

done

python3 - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/metrics.json")):
    data = json.loads(path.read_text())
    rows.append({
        "arm": path.parents[1].name,
        "pp512": data.get("pp512"),
        "tg128": data.get("tg128"),
        "metrics": str(path),
        "quality_safe": path.parents[1].name == "full",
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(root / "summary.json")
PY

printf '\nWARNING: rows-* arms mask uncomputed logits to -inf and are not quality-safe.\n'
