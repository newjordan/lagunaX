#!/usr/bin/env python3
"""Audit GPU-layer offload coverage in Laguna benchmark artifacts."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BENCH = ROOT / "scripts" / "bench-serial.sh"
ENV = ROOT / "env.sh"

values = []
parsed = 0
for path in RESULTS.rglob("*.json"):
    try:
        obj = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        continue
    parsed += 1
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                normalized = key.lower().replace("-", "_")
                if normalized in {"n_gpu_layers", "ngl", "gpu_layers"}:
                    if isinstance(value, (int, float, str)) and str(value).lstrip("-").isdigit():
                        values.append(int(value))
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)

env_text = ENV.read_text()
bench_text = BENCH.read_text()
default = re.search(r'export NGL="\$\{NGL:-(-?\d+)\}"', env_text)
if not default:
    raise SystemExit("NGL default not found")
if '-ngl "$NGL"' not in bench_text:
    raise SystemExit("serial harness does not pass NGL")

counts = Counter(values)
print(f"json_artifacts_parsed={parsed}")
print(f"gpu_layer_records={len(values)}")
print("gpu_layer_value_counts=" + json.dumps(dict(sorted(counts.items()))))
print(f"active_ngl_default={default.group(1)}")
print("serial_harness_passes_ngl=true")
print(f"distinct_gpu_layer_values={len(counts)}")
