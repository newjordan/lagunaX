#!/usr/bin/env python3
"""Audit kernel-level profiling coverage in Laguna result artifacts."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = [ROOT / "results", ROOT / "benchmark" / "results"]
files = sorted({p for r in RESULT_ROOTS if r.exists() for p in r.rglob("*") if p.is_file()})
trace_named = [p for p in files if any(s in p.name.lower() for s in ("ktrace", "kernel-trace", "unitrace"))]
profile_named = [p for p in files if "profile" in p.name.lower()]
terms = {
    "kernel_duration": ("kernel duration", "kernel_time", "device time", "gpu time"),
    "kernel_name": ("kernel name", "kernel_name"),
    "occupancy": ("occupancy",),
    "eu_active": ("eu active", "eu_active"),
    "memory_bandwidth": ("memory bandwidth", "bandwidth gb"),
}
hits = {k: [] for k in terms}
for p in files:
    try:
        text = p.read_text(errors="ignore").lower()
    except OSError:
        continue
    for key, needles in terms.items():
        if any(n in text for n in needles):
            hits[key].append(str(p.relative_to(ROOT)))

out = {
    "files_scanned": len(files),
    "trace_named_files": len(trace_named),
    "profile_named_files": len(profile_named),
    "coverage": {k: len(v) for k, v in hits.items()},
    "trace_paths": [str(p.relative_to(ROOT)) for p in trace_named[:20]],
}
out_path = ROOT / "benchmark" / "results" / "kernel-critical-path-audit-20260807.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, sort_keys=True))
