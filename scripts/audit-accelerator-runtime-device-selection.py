#!/usr/bin/env python3
"""Audit accelerator runtime device-selection controls and benchmark provenance."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
keys = ("ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER", "ZE_AFFINITY_MASK")

def source_value(key: str):
    match = re.search(rf"export\s+{key}=['\"]?([^'\"\s]+)", env_text)
    if not match:
        return None
    value = match.group(1)
    default = re.fullmatch(rf"\$\{{{key}:-([^}}]+)}}", value)
    return default.group(1) if default else value

payload = {
    "angle": "accelerator_runtime_device_selection",
    "active_environment": {key: os.environ.get(key) for key in keys},
    "canonical_environment": {key: source_value(key) for key in keys},
    "harness_mentions": {key: key in bench_text for key in keys},
    "metrics_record": {key: bool(re.search(rf'["\']{key}["\']', bench_text)) for key in keys},
}
out = ROOT / "results/accelerator-runtime-device-selection-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
