#!/usr/bin/env bash
# Controlled batch/microbatch sweep using the existing serial benchmark contract.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

BATCH_SWEEP="${BATCH_SWEEP:-512 1024 2048}"
UBATCH_SWEEP="${UBATCH_SWEEP:-512 1024 2048}"
SWEEP_ROOT="${SWEEP_ROOT:-$LX_RESULTS/batch-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$SWEEP_ROOT"

for batch in $BATCH_SWEEP; do
    for ubatch in $UBATCH_SWEEP; do
        if (( ubatch > batch )); then
            continue
        fi
        run_dir="$SWEEP_ROOT/b${batch}-ub${ubatch}"
        echo "== batch=$batch ubatch=$ubatch =="
        BBATCH="$batch" UBATCH="$ubatch" LX_RESULTS="$run_dir" \
            "$ROOT/scripts/bench-serial.sh" candidate "batch=$batch ubatch=$ubatch"
    done
done

python3 - "$SWEEP_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in root.glob("**/metrics.json"):
    record = json.loads(path.read_text())
    rows.append({
        "batch": record["flags"]["bbatch"],
        "ubatch": record["flags"]["ubatch"],
        "pp512": record["pp512"],
        "tg128": record["tg128"],
        "metrics": str(path),
    })
rows.sort(key=lambda row: (row["batch"], row["ubatch"]))
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
if rows:
    print("best pp512:", max(rows, key=lambda row: row["pp512"]))
    print("best tg128:", max(rows, key=lambda row: row["tg128"]))
PY
