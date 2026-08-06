#!/usr/bin/env python3
"""Audit dynamic-loader and accelerator-layer injection in active Laguna benchmarks."""
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "runtime-injection-policy-audit-20260807.json"
SOURCES = [ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh"]
CONTROLS = (
    "LD_PRELOAD",
    "LD_AUDIT",
    "ZE_ENABLE_LOADER_DEBUG_TRACE",
    "ZE_ENABLE_VALIDATION_LAYER",
    "ZE_ENABLE_TRACING_LAYER",
    "SYCL_PI_TRACE",
)

source_mentions = {
    str(path.relative_to(ROOT)): [key for key in CONTROLS if key in path.read_text()]
    for path in SOURCES
}
artifact_mentions = {key: [] for key in CONTROLS}
parsed = 0
for path in ROOT.rglob("*.json"):
    if path == OUT:
        continue
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    text = json.dumps(data, sort_keys=True)
    for key in CONTROLS:
        if key in text:
            artifact_mentions[key].append(str(path.relative_to(ROOT)))

report = {
    "audit": "runtime-injection-policy",
    "active_sources": source_mentions,
    "process_environment": {key: os.environ.get(key) for key in CONTROLS},
    "effective_injection_active": any(os.environ.get(key) for key in CONTROLS),
    "artifacts": {"json_files_parsed": parsed, "mentions": artifact_mentions},
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
