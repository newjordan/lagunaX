#!/usr/bin/env python3
"""Audit benchmark repetition-depth coverage and reconcile the literal score target."""
import json
from collections import Counter
from pathlib import Path

OUT = Path("results/repetition-depth-audit-20260807.json")
counts = Counter()
parsed_files = 0
records = 0


def visit(value):
    global records
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"reps", "repetitions", "n_repetitions"} and not isinstance(item, (dict, list, bool)):
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
        text = path.read_text(errors="replace")
        values = []
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

score = json.loads(Path("results/LATEST_SCORE.json").read_text())["score"]
report = {
    "parsed_artifact_files": parsed_files,
    "records_with_repetition_depth": records,
    "repetition_depth_counts": dict(sorted(counts.items())),
    "active_default_repetitions": 5,
    "serial_harness_passes_repetitions": True,
    "literal_target_score": 2.0,
    "latest_verified_score": score,
    "literal_target_delta": 2.0 - score,
}
assert parsed_files > 0
assert score < 2.0
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
