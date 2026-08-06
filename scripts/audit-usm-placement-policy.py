#!/usr/bin/env python3
"""Audit explicit SYCL/Level Zero unified-memory placement and prefetch policy."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh"]
CONTROLS = (
    "SYCL_PI_LEVEL_ZERO_USM_RESIDENT",
    "SYCL_PI_LEVEL_ZERO_USM_RESIDENT_DEVICE",
    "SYCL_PI_LEVEL_ZERO_USM_RESIDENT_SHARED",
    "SYCL_PI_LEVEL_ZERO_USM_RESIDENT_HOST",
    "SYCL_PI_LEVEL_ZERO_USE_USM_ALLOCATOR",
    "SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR",
    "ZE_ENABLE_PCI_ID_DEVICE_ORDER",
)

source_hits = {name: [] for name in CONTROLS}
for path in SOURCES:
    for number, line in enumerate(path.read_text().splitlines(), 1):
        for name in CONTROLS:
            if name in line:
                source_hits[name].append(f"{path.relative_to(ROOT)}:{number}")

environment = {name: os.environ.get(name) for name in CONTROLS}
artifact_hits = {name: 0 for name in CONTROLS}
parsed = 0
for path in ROOT.glob("results/**/*.json"):
    if path.name == "usm-placement-policy-audit-20260807.json":
        continue
    try:
        text = path.read_text(errors="replace")
        json.loads(text)
    except (OSError, json.JSONDecodeError):
        continue
    parsed += 1
    for name in CONTROLS:
        artifact_hits[name] += text.count(name)

out = {
    "angle": "SYCL/Level Zero unified-memory residency and allocator policy",
    "controls": list(CONTROLS),
    "source_hits": source_hits,
    "environment": environment,
    "parsed_json_artifacts": parsed,
    "artifact_mentions": artifact_hits,
    "explicit_active_controls": sum(bool(v) for v in source_hits.values())
    + sum(v is not None for v in environment.values()),
}
dest = ROOT / "results" / "usm-placement-policy-audit-20260807.json"
dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(dest)
