#!/usr/bin/env python3
"""Audit host allocator policy and historical coverage for Laguna benchmarks."""
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "env.sh", ROOT / "scripts/bench-serial.sh"]
PATTERNS = {
    "ld_preload": r"\bLD_PRELOAD\b",
    "jemalloc": r"\bjemalloc\b|\bMALLOC_CONF\b",
    "tcmalloc": r"\btcmalloc\b|\bTCMALLOC_[A-Z_]+\b",
    "mimalloc": r"\bmimalloc\b|\bMIMALLOC_[A-Z_]+\b",
    "glibc_malloc_tuning": r"\bMALLOC_(?:ARENA_MAX|MMAP_THRESHOLD_|TRIM_THRESHOLD_)\b|\bGLIBC_TUNABLES\b",
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
        if path.name == "host-allocator-policy-audit-20260807.json":
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
                    if any(token in normalized for token in ("allocator", "jemalloc", "tcmalloc", "mimalloc", "malloc_conf", "ld_preload")):
                        keys[normalized] += 1
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
report = {
    "audit": "host-allocator-policy",
    "active_source_hits": source_hits,
    "active_explicit_controls": sum(bool(v) for v in source_hits.values()),
    "process_environment": {key: os.environ.get(key) for key in ("LD_PRELOAD", "MALLOC_CONF", "GLIBC_TUNABLES", "MALLOC_ARENA_MAX")},
    "json_artifacts_parsed": parsed,
    "historical_allocator_keys": dict(sorted(keys.items())),
    "historical_allocator_records": sum(keys.values()),
}
out = ROOT / "benchmark/results/host-allocator-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
