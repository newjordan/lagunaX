#!/usr/bin/env python3
"""Audit batch and microbatch policy coverage in Laguna result artifacts."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "batch-size-coverage-audit-20260807.json"

batch_keys = {"batch", "bbatch", "n_batch"}
ubatch_keys = {"ubatch", "microbatch", "n_ubatch"}
batches = Counter()
ubatches = Counter()
parsed = 0

def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in batch_keys and isinstance(child, (int, float)) and not isinstance(child, bool):
                batches[str(int(child))] += 1
            if normalized in ubatch_keys and isinstance(child, (int, float)) and not isinstance(child, bool):
                ubatches[str(int(child))] += 1
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

for path in RESULTS.rglob("*.json"):
    if path == OUT:
        continue
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    walk(payload)

report = {
    "json_artifacts_parsed": parsed,
    "batch_records": sum(batches.values()),
    "batch_values": dict(sorted(batches.items(), key=lambda item: int(item[0]))),
    "ubatch_records": sum(ubatches.values()),
    "ubatch_values": dict(sorted(ubatches.items(), key=lambda item: int(item[0]))),
    "distinct_batch_values": len(batches),
    "distinct_ubatch_values": len(ubatches),
}
assert parsed > 0
assert report["batch_records"] > 0
assert report["ubatch_records"] > 0
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
