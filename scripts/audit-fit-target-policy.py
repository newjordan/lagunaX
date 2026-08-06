#!/usr/bin/env python3
"""Audit automatic device-memory fitting policy and historical provenance."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()
OUT = ROOT / "results/fit-target-policy-audit-20260807.json"

cmd = f"source {ROOT / 'env.sh'} >/dev/null 2>&1; \"$LX_LLAMA_BENCH\" --help"
help_text = subprocess.run(["bash", "-c", cmd], text=True, capture_output=True, check=True).stdout
match = re.search(r"--fit-target <MiB>.*?\(default: ([^)]+)\)", help_text)

mentions = []
values = []
parsed = 0
for path in sorted(ROOT.rglob("*.json")):
    if path == OUT:
        continue
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() == "fit_target":
                    found.append(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(data)
    if found:
        mentions.append(str(path.relative_to(ROOT)))
        values.extend(found)

source_explicit = bool(re.search(r"(?:--fit-target|fit_target|FIT_TARGET)", ENV + "\n" + BENCH))
payload = {
    "control": "--fit-target <MiB>",
    "executable_default": match.group(1) if match else None,
    "active_source_explicit": source_explicit,
    "parsed_json_artifacts": parsed,
    "artifact_mentions": mentions,
    "artifact_value_counts": {str(value): values.count(value) for value in sorted(set(values), key=str)},
    "environment_value": os.environ.get("FIT_TARGET"),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(OUT)
