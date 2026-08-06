#!/usr/bin/env python3
"""Audit active flash-attention policy and reconcile the literal 2x target."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
score = json.loads((root / "results/LATEST_SCORE.json").read_text())

fa = re.search(r'export FA="\$\{FA:-(?P<value>-?\d+)\}"', env)
if not fa:
    raise SystemExit("cannot derive FA default")
assert '-fa "$FA"' in bench
assert 'CTX="${CTX:-8192}"' in env
# CTX is provenance-only: it appears in emitted metadata, not COMMON or either invocation.
common = bench.split("COMMON=(", 1)[1].split(")", 1)[0]
assert '"$CTX"' not in common
assert re.search(r'"\$LX_LLAMA_BENCH" "\$\{COMMON\[@\]\}" -p "\$LX_PP" -n 0', bench)
assert re.search(r'"\$LX_LLAMA_BENCH" "\$\{COMMON\[@\]\}" -p 0 -n "\$LX_TG"', bench)

current = float(score["score"])
artifact = {
    "audit": "flash-attention-and-target-reconciliation",
    "active_policy": {
        "flash_attention_value": int(fa.group("value")),
        "flash_attention_semantics": "auto",
        "harness_passes_flash_attention_explicitly": True,
    },
    "evidence_reconciliation": {
        "latest_floors_ok": bool(score["floors_ok"]),
        "latest_score": current,
        "literal_target": 2.0,
        "absolute_score_delta": 2.0 - current,
        "relative_improvement_required_from_current_pct": (2.0 / current - 1.0) * 100.0,
        "latest_decode_speedup": score["decode_speedup"],
        "latest_prefill_speedup": score["prefill_speedup"],
        "ctx_exported_but_not_consumed_by_serial_harness": True,
        "known_negative_results": [
            "any-batch MUL_MAT_ADD prefill wrecks wikitext perplexity",
            "QKV shared quant produced a device-lost probe",
        ],
    },
}
out = root / "results/flash-attention-target-reconciliation-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
