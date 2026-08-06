#!/usr/bin/env python3
"""Audit host thread-count and CPU-affinity provenance without running the GPU."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bench = (root / "scripts/bench-serial.sh").read_text()
env = (root / "env.sh").read_text()
score_path = root / "results/LATEST_SCORE.json"
score = json.loads(score_path.read_text())

controls = ("taskset", "numactl", "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "OMP_PROC_BIND", "OMP_PLACES")
report = {
    "audit": "cpu-thread-affinity-policy",
    "sources": {
        "thread_default": 16 if 'export THREADS="${THREADS:-16}"' in env else None,
        "bench_passes_threads": '-t "$THREADS"' in bench,
        "explicit_affinity_controls": {key: (key in env or key in bench) for key in controls},
        "logical_cpu_count": os.cpu_count(),
    },
    "target_reconciliation": {
        "literal_target_score": 2.0,
        "latest_score": score["score"],
        "score_delta_remaining": 2.0 - score["score"],
        "latest_increase_pct": score["increase_pct"],
        "target_increase_pct": 100.0,
        "percentage_point_delta_remaining": 100.0 - score["increase_pct"],
        "floors_ok": score["floors_ok"],
    },
    "benchmark_launched": False,
}
out = root / "results/cpu-thread-affinity-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
