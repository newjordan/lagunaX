#!/usr/bin/env python3
"""Audit llama-bench host polling policy coverage."""
import json
from collections import Counter
from pathlib import Path

OUT = Path("results/device-poll-policy-audit-20260807.json")
ENV = Path("env.sh").read_text(errors="replace")
BENCH = Path("scripts/bench-serial.sh").read_text(errors="replace")
counts = Counter()
parsed_files = 0
records = 0
KEYS = {"poll", "poll_percent", "device_poll", "device_poll_percent"}


def visit(value):
    global records
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in KEYS:
                counts[str(item)] += 1
                records += 1
            visit(item)
    elif isinstance(value, list):
        for item in value:
            visit(item)


for root in (Path("baseline"), Path("results")):
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if path == OUT or not path.is_file() or path.suffix not in {".json", ".log"}:
            continue
        values = []
        text = path.read_text(errors="replace")
        try:
            values = [json.loads(text)]
        except json.JSONDecodeError:
            for line in text.splitlines():
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if values:
            parsed_files += 1
            for value in values:
                visit(value)

report = {
    "control": "llama-bench --poll <0...100>",
    "active_binary_help_default": 50,
    "env_declares_poll": "POLL=" in ENV or "export POLL" in ENV,
    "serial_harness_passes_poll": "--poll" in BENCH,
    "parsed_artifact_files": parsed_files,
    "artifact_poll_records": records,
    "artifact_poll_value_counts": dict(sorted(counts.items())),
}
assert parsed_files > 0
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
