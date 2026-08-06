#!/usr/bin/env python3
"""Audit generation sampling-policy provenance in quality/proof paths and artifacts."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "scripts/proof-suite.sh", ROOT / "scripts/laguna-ab-suite.sh"]
FIELDS = ("temperature", "top_k", "top_p", "min_p", "seed", "samplers")
patterns = {
    field: re.compile(rf"(?:--{field.replace('_', '-')}\b|[\"']{field}[\"']\s*:)")
    for field in FIELDS
}
source_hits = {}
for path in SOURCES:
    text = path.read_text(errors="replace")
    source_hits[str(path.relative_to(ROOT))] = {
        field: len(pattern.findall(text)) for field, pattern in patterns.items()
    }

artifact_counts = {field: 0 for field in FIELDS}
json_files = 0
for path in (ROOT / "results").rglob("*.json"):
    if "sampling-policy-audit" in path.name:
        continue
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        continue
    json_files += 1
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                normalized = key.lower().replace("-", "_")
                if normalized in artifact_counts:
                    artifact_counts[normalized] += 1
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)

report = {
    "angle": "generation sampling-policy provenance",
    "sources": source_hits,
    "artifact_json_files": json_files,
    "artifact_field_occurrences": artifact_counts,
    "finding": "temperature and seed are explicit in proof paths; the rest of the sampling chain is not fully pinned or preserved",
}
out = ROOT / "results" / "sampling-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
