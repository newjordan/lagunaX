#!/usr/bin/env python3
"""Audit explicit Level Zero copy/compute engine-routing policy coverage."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCES = (ROOT / "env.sh", ROOT / "scripts" / "bench-serial.sh")
RESULTS = ROOT / "results"
KNOBS = (
    "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE",
    "SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE",
    "UR_L0_USE_COPY_ENGINE",
    "UR_L0_USE_COMPUTE_ENGINE",
)
REPORT_PREFIX = "level-zero-engine-routing-audit-"

sources = {
    str(path.relative_to(ROOT)): path.read_text(errors="replace")
    for path in ACTIVE_SOURCES
}
artifacts: list[tuple[str, str]] = []
for path in RESULTS.rglob("*"):
    if path.name.startswith(REPORT_PREFIX):
        continue
    if path.is_file() and path.suffix.lower() in {".json", ".txt", ".log", ".md"}:
        try:
            artifacts.append((str(path.relative_to(ROOT)), path.read_text(errors="replace")))
        except OSError:
            pass

report = {
    "policy": "level_zero_copy_compute_engine_routing",
    "knobs": list(KNOBS),
    "source_explicit": {
        knob: [name for name, text in sources.items() if knob in text] for knob in KNOBS
    },
    "process_environment": {knob: os.environ.get(knob) for knob in KNOBS},
    "artifact_explicit_counts": {
        knob: sum(knob in text for _, text in artifacts) for knob in KNOBS
    },
    "artifacts_scanned": len(artifacts),
}
report["active_policy_explicit"] = any(report["source_explicit"].values()) or any(
    value is not None for value in report["process_environment"].values()
)
report["artifact_policy_preserved"] = any(report["artifact_explicit_counts"].values())
print(json.dumps(report, indent=2, sort_keys=True))
