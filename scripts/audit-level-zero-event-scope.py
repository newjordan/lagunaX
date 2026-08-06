#!/usr/bin/env python3
"""Audit Level Zero event/synchronization-scope policy coverage."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = (
    "SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS",
    "UR_L0_USE_IMMEDIATE_COMMANDLISTS",
    "UR_L0_USE_DRIVER_INORDER_LISTS",
    "UR_L0_USE_COPY_ENGINE",
)
sources = {p.name: p.read_text(errors="replace") for p in (ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh")}
source_mentions = {key: [name for name, text in sources.items() if key in text] for key in CONTROLS}
environment = {key: os.environ.get(key) for key in CONTROLS}
artifact_mentions = {key: 0 for key in CONTROLS}
parsed = 0
out = ROOT / "results" / "level-zero-event-scope-audit-20260807.json"
for path in ROOT.rglob("*.json"):
    if path == out:
        continue
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        continue
    parsed += 1
    text = json.dumps(data)
    for key in CONTROLS:
        artifact_mentions[key] += text.count(key)
report = {
    "controls": list(CONTROLS),
    "source_mentions": source_mentions,
    "environment": environment,
    "parsed_json_artifacts": parsed,
    "artifact_mentions": artifact_mentions,
}
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
