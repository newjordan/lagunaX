#!/usr/bin/env python3
"""Audit active llama-bench host worker-thread policy and provenance."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()

m = re.search(r'export THREADS="\$\{THREADS:-([0-9]+)\}"', env_text)
assert m, "THREADS default missing from env.sh"
default_threads = int(m.group(1))
explicit_pass = bool(re.search(r'-t\s+"\$THREADS"', bench_text))
assert explicit_pass, "bench-serial.sh does not explicitly pass THREADS"
effective_threads = int(os.environ.get("THREADS", default_threads))
logical_cpus = os.cpu_count() or 0

artifacts = []
recorded = []
for path in sorted(ROOT.glob("**/*.json")):
    if any(part == ".git" for part in path.parts):
        continue
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    artifacts.append(str(path.relative_to(ROOT)))
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "threads" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    recorded.append({"artifact": str(path.relative_to(ROOT)), "value": int(value)})
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)

payload = {
    "policy": {
        "env_default_threads": default_threads,
        "process_override": os.environ.get("THREADS"),
        "effective_threads": effective_threads,
        "bench_explicitly_passes_threads": explicit_pass,
        "logical_cpu_count": logical_cpus,
        "effective_fraction_of_logical_cpus": effective_threads / logical_cpus if logical_cpus else None,
    },
    "historical_coverage": {
        "json_artifacts_parsed": len(artifacts),
        "thread_values_recorded": len(recorded),
        "distinct_thread_values": sorted({r["value"] for r in recorded}),
    },
}
out = ROOT / "results" / "host-worker-thread-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
print(json.dumps(payload, indent=2))
