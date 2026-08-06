#!/usr/bin/env python3
"""Audit power, clock, temperature, and throttling provenance for Laguna benchmarks."""
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "env.sh", ROOT / "scripts/bench-serial.sh"]
PATTERNS = {
    "gpu_power": r"\b(?:power_limit|power_draw|gpu_power)\b",
    "gpu_clock": r"\b(?:gpu_clock|graphics_clock|memory_clock|frequency)\b",
    "temperature": r"\b(?:gpu_temperature|temperature|thermal)\b",
    "throttling": r"\b(?:throttl(?:e|ing)|performance_limit)\b",
}
source_hits = {
    name: [str(path.relative_to(ROOT)) for path in SOURCES if re.search(pattern, path.read_text(), re.I)]
    for name, pattern in PATTERNS.items()
}
keys = Counter()
parsed = 0
for base in (ROOT / "results", ROOT / "benchmark/results"):
    if not base.exists():
        continue
    for path in base.rglob("*.json"):
        if path.name == "power-clock-thermal-provenance-audit-20260807.json":
            continue
        try:
            obj = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        parsed += 1
        stack = [obj]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    normalized = key.lower()
                    if any(token in normalized for token in ("power", "clock", "frequency", "temperature", "thermal", "throttl")):
                        keys[normalized] += 1
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)

def capture(command):
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return {"command": command, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}

report = {
    "audit": "power-clock-thermal-provenance",
    "active_source_hits": source_hits,
    "active_explicit_controls": sum(bool(v) for v in source_hits.values()),
    "environment_matches": sorted(key for key in os.environ if re.search(r"power|clock|freq|thermal|throttl", key, re.I)),
    "device_snapshot": capture(["xpu-smi", "discovery", "-d", "0"]),
    "json_artifacts_parsed": parsed,
    "historical_provenance_keys": dict(sorted(keys.items())),
    "historical_provenance_records": sum(keys.values()),
}
out = ROOT / "benchmark/results/power-clock-thermal-provenance-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
