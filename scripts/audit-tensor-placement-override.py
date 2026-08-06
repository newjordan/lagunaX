#!/usr/bin/env python3
"""Audit explicit tensor-placement override coverage in Laguna benchmarks/results."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh"]
NEEDLES = ("--override-tensor", "-ot", "override_tensor")

source_hits = {}
for path in SOURCES:
    text = path.read_text(errors="replace")
    source_hits[str(path.relative_to(ROOT))] = [n for n in NEEDLES if n in text]

artifact_files = 0
artifact_mentions = []
out = ROOT / "results" / "tensor-placement-override-audit-20260807.json"
for path in sorted((ROOT / "results").rglob("*.json")):
    if path == out:
        continue
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        continue
    artifact_files += 1
    blob = json.dumps(data, sort_keys=True).lower()
    hits = [n for n in NEEDLES if n.lower() in blob]
    if hits:
        artifact_mentions.append({"path": str(path.relative_to(ROOT)), "needles": hits})

report = {
    "policy": "tensor placement override",
    "executable_help": {
        "flag": "-ot/--override-tensor",
        "syntax": "<tensor name pattern>=<buffer type>;...",
        "default": "disabled",
    },
    "active_source_hits": source_hits,
    "process_environment": {
        k: v for k, v in os.environ.items() if "OVERRIDE_TENSOR" in k.upper()
    },
    "parsed_json_artifacts": artifact_files,
    "artifact_mentions": artifact_mentions,
}
out = ROOT / "results" / "tensor-placement-override-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
