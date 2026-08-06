#!/usr/bin/env python3
"""Audit whether the active serial benchmark uses llama-bench's combined -pg mode."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "bench-serial.sh"
RESULTS = ROOT / "results"
OUT = RESULTS / "combined-prompt-generation-audit-20260807.json"

source = BENCH.read_text()
invocations = re.findall(r'PP_JSON=.*?\n|TG_JSON=.*?\n', source)
artifacts = 0
pg_records = []
for path in RESULTS.rglob("*.json"):
    if path == OUT:
        continue
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    artifacts += 1
    text = json.dumps(value, sort_keys=True)
    if re.search(r'"(?:n_)?prompt"\s*:\s*[1-9]\d*', text) and re.search(r'"(?:n_)?gen"\s*:\s*[1-9]\d*', text):
        pg_records.append(str(path.relative_to(ROOT)))

payload = {
    "active_benchmark": str(BENCH.relative_to(ROOT)),
    "llama_bench_process_invocations": len(invocations),
    "active_uses_combined_pg_flag": bool(re.search(r'(^|\s)-pg(\s|$)', source)),
    "active_prefill_and_decode_are_separate_processes": len(invocations) == 2,
    "json_artifacts_parsed": artifacts,
    "artifacts_with_nonzero_prompt_and_generation_fields": pg_records,
    "artifacts_with_nonzero_prompt_and_generation_count": len(pg_records),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
