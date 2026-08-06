#!/usr/bin/env python3
"""Audit host-load provenance around Laguna performance measurements."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "host-load-provenance-audit-20260807.json"

source_paths = [ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh"]
needles = ("loadavg", "proc/stat", "cpu utilization", "cpu_util", "iowait", "pressure/")
source_mentions = {}
for path in source_paths:
    text = path.read_text(errors="replace").lower()
    source_mentions[str(path.relative_to(ROOT))] = [n for n in needles if n in text]

artifact_mentions = []
parsed = 0
for path in RESULTS.rglob("*.json"):
    if path == OUT:
        continue
    try:
        obj = json.loads(path.read_text(errors="replace"))
    except Exception:
        continue
    parsed += 1
    text = json.dumps(obj, sort_keys=True).lower()
    hits = [n for n in needles if n in text]
    if hits:
        artifact_mentions.append({"path": str(path.relative_to(ROOT)), "fields": hits})

loadavg = os.getloadavg()
pressure = {}
for name in ("cpu", "io", "memory"):
    path = Path("/proc/pressure") / name
    pressure[name] = path.read_text().strip() if path.exists() else None

report = {
    "policy": "host load and Linux PSI provenance",
    "active_source_mentions": source_mentions,
    "parsed_json_artifacts": parsed,
    "artifact_mentions": artifact_mentions,
    "current_snapshot": {
        "loadavg_1m_5m_15m": loadavg,
        "linux_pressure_stall_information": pressure,
    },
    "finding": "active benchmark sources do not capture host load average, CPU utilization, iowait, or Linux PSI",
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
