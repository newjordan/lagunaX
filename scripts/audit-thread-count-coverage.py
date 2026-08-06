#!/usr/bin/env python3
"""Audit host thread-count coverage in Laguna benchmark artifacts."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = [ROOT / "results", ROOT / "baseline", ROOT / "benchmark" / "results"]
THREAD_KEYS = {"threads", "n_threads", "n_threads_main", "thread_count"}

files = records = explicit = 0
values: Counter[int] = Counter()
for base in RESULT_ROOTS:
    if not base.exists():
        continue
    for path in base.rglob("*.json"):
        if path.name.startswith("thread-count-coverage-audit-"):
            continue
        files += 1
        try:
            obj = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        stack = [obj]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                records += 1
                for key, value in node.items():
                    if key.lower() in THREAD_KEYS:
                        explicit += 1
                        try:
                            values[int(value)] += 1
                        except (TypeError, ValueError):
                            pass
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)

serial = (ROOT / "scripts" / "bench-serial.sh").read_text()
env = (ROOT / "env.sh").read_text()
assert '-t "$THREADS"' in serial
assert 'export THREADS="${THREADS:-16}"' in env
assert files > 0 and records > 0

out = {
    "json_files_scanned": files,
    "dictionary_records_scanned": records,
    "explicit_thread_fields": explicit,
    "thread_values": dict(sorted(values.items())),
    "serial_thread_control": "THREADS",
    "serial_default_threads": 16,
    "serial_passes_thread_flag": True,
    "distinct_recorded_thread_counts": len(values),
}
print(json.dumps(out, indent=2, sort_keys=True))
