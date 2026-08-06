#!/usr/bin/env python3
"""Audit coverage of quality-sensitive SYCL fusion controls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "GGML_SYCL_DISABLE_MOE_DUAL_DOWN",
    "GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN",
    "GGML_SYCL_DISABLE_QKV_SHARED_QUANT",
)
env = (ROOT / "env.sh").read_text()
active = {key: key in env for key in KEYS}
counts = {key: {"0": 0, "1": 0} for key in KEYS}
parsed = 0
for path in ROOT.rglob("*.json"):
    if path.name == "fusion-policy-audit-20260807.json":
        continue
    try:
        text = path.read_text(errors="replace")
        json.loads(text)
        parsed += 1
    except (OSError, json.JSONDecodeError):
        continue
    for key in KEYS:
        for value in ("0", "1"):
            counts[key][value] += text.count(f'"{key}": "{value}"')
            counts[key][value] += text.count(f'"{key}": {value}')
report = {"parsed_json_artifacts": parsed, "active_env_mentions": active, "artifact_value_counts": counts}
out = ROOT / "results" / "fusion-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
