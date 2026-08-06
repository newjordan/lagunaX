#!/usr/bin/env python3
"""Audit host math-runtime thread-pool controls that can contend with GPU submission."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh"]
CONTROLS = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "DNNL_MAX_CPU_ISA",
    "KMP_BLOCKTIME",
    "KMP_LIBRARY",
)

def mentions(text: str) -> dict[str, bool]:
    return {key: key in text for key in CONTROLS}

source_mentions = {
    str(path.relative_to(ROOT)): mentions(path.read_text()) for path in SOURCES
}
environment = {key: os.environ.get(key) for key in CONTROLS}
artifact_files = []
artifact_mentions = {key: 0 for key in CONTROLS}
for path in sorted(ROOT.rglob("*.json")):
    if path.name == "host-math-threadpool-policy-audit-20260807.json":
        continue
    try:
        text = path.read_text(errors="replace")
        json.loads(text)
    except (OSError, json.JSONDecodeError):
        continue
    artifact_files.append(str(path.relative_to(ROOT)))
    for key, present in mentions(text).items():
        artifact_mentions[key] += int(present)

report = {
    "policy": "host_math_threadpool",
    "controls": list(CONTROLS),
    "source_mentions": source_mentions,
    "environment": environment,
    "artifact_json_files_parsed": len(artifact_files),
    "artifact_mentions": artifact_mentions,
}
print(json.dumps(report, indent=2, sort_keys=True))
