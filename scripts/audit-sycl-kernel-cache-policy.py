#!/usr/bin/env python3
"""Audit SYCL persistent kernel-cache policy and provenance."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "sycl-kernel-cache-policy-audit-20260807.json"
CONTROLS = ("SYCL_CACHE_PERSISTENT", "SYCL_CACHE_DIR", "SYCL_CACHE_THRESHOLD", "SYCL_CACHE_MAX_SIZE")

active = {key: os.environ.get(key) for key in CONTROLS}
source_text = "\n".join(
    path.read_text(errors="replace")
    for path in (ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh")
)
source_mentions = {key: key in source_text for key in CONTROLS}
artifact_mentions = {key: 0 for key in CONTROLS}
parsed_json = 0
for path in RESULTS.rglob("*.json"):
    if path == OUT:
        continue
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        continue
    parsed_json += 1
    text = json.dumps(data)
    for key in CONTROLS:
        artifact_mentions[key] += text.count(key)

payload = {
    "angle": "SYCL persistent kernel compilation cache policy",
    "controls": list(CONTROLS),
    "active_environment": active,
    "active_source_mentions": source_mentions,
    "parsed_result_json": parsed_json,
    "artifact_mentions": artifact_mentions,
    "explicit_active_controls": sum(value is not None for value in active.values()),
    "source_control_count": sum(source_mentions.values()),
    "artifact_control_mentions": sum(artifact_mentions.values()),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
assert parsed_json > 0
assert payload["explicit_active_controls"] == 0
assert payload["source_control_count"] == 0
assert payload["artifact_control_mentions"] == 0
print(OUT)
