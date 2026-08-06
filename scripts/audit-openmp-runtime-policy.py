#!/usr/bin/env python3
"""Audit host OpenMP runtime policy provenance for canonical Laguna benchmarks."""
import json
import os
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
source = env_text + "\n" + bench_text
keys = [
    "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS", "OMP_NESTED", "OMP_SCHEDULE",
    "OMP_THREAD_LIMIT", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME",
]
configured = {key: bool(re.search(rf"(?<![A-Za-z0-9_]){key}(?![A-Za-z0-9_])", source)) for key in keys}
effective = {key: os.environ.get(key) for key in keys}
recorded = {key: bool(re.search(rf'["\']{key.lower()}["\']\s*:', bench_text, re.I)) for key in keys}
artifact = {
    "audit": "openmp-runtime-policy",
    "controls": keys,
    "canonical_configured": configured,
    "effective_environment": effective,
    "metrics_recorded": recorded,
    "canonical_controls_any": any(configured.values()),
    "effective_overrides_any": any(value is not None for value in effective.values()),
    "metrics_record_any": any(recorded.values()),
}
out = root / "results/openmp-runtime-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(out)
