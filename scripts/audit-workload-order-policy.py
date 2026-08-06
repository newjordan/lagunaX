#!/usr/bin/env python3
"""Audit serial benchmark workload-order policy and historical order coverage."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "bench-serial.sh"
text = BENCH.read_text()

pp_pos = text.index('PP_JSON="$(')
tg_pos = text.index('TG_JSON="$(')
randomization_tokens = re.findall(
    r"(?im)^.*(?:shuf|RANDOM|random|alternate|counterbalance|workload.order).*$", text
)

artifacts = sorted((ROOT / "results").rglob("*.json"))
order_fields = 0
parsed = 0
for path in artifacts:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    blob = json.dumps(payload).lower()
    if any(key in blob for key in ('"workload_order"', '"test_order"', '"run_order"')):
        order_fields += 1

report = {
    "policy": {
        "prefill_source_offset": pp_pos,
        "decode_source_offset": tg_pos,
        "fixed_order": "prefill_then_decode" if pp_pos < tg_pos else "decode_then_prefill",
        "order_randomized_or_counterbalanced": bool(randomization_tokens),
        "matching_source_lines": randomization_tokens,
        "separate_processes": text.count('"$LX_LLAMA_BENCH" "${COMMON[@]}"') == 2,
    },
    "historical_coverage": {
        "json_artifacts_parsed": parsed,
        "artifacts_recording_order_field": order_fields,
    },
}
out = ROOT / "results" / "workload-order-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
assert report["policy"]["fixed_order"] == "prefill_then_decode"
assert not report["policy"]["order_randomized_or_counterbalanced"]
assert report["policy"]["separate_processes"]
print(out)
