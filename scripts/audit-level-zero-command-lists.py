#!/usr/bin/env python3
"""Audit explicit Level Zero immediate-command-list policy coverage."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
BENCH = ROOT / "scripts" / "bench-serial.sh"
RESULTS = ROOT / "results"
KNOBS = (
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    "UR_L0_USE_IMMEDIATE_COMMANDLISTS",
    "ZE_ENABLE_IMMEDIATE_COMMANDLISTS",
)

sources = {str(path.relative_to(ROOT)): path.read_text(errors="replace") for path in (ENV, BENCH)}
artifacts = []
for path in RESULTS.rglob("*"):
    if path.name.startswith("level-zero-command-list-policy-audit-"):
        continue
    if path.is_file() and path.suffix.lower() in {".json", ".txt", ".log", ".md"}:
        try:
            artifacts.append((str(path.relative_to(ROOT)), path.read_text(errors="replace")))
        except OSError:
            pass

report = {
    "policy": "level_zero_immediate_command_lists",
    "knobs": list(KNOBS),
    "source_explicit": {knob: [name for name, text in sources.items() if knob in text] for knob in KNOBS},
    "artifact_explicit_counts": {
        knob: sum(knob in text for _, text in artifacts) for knob in KNOBS
    },
    "artifacts_scanned": len(artifacts),
    "process_environment": {knob: os.environ.get(knob) for knob in KNOBS},
}
report["serial_policy_explicit"] = any(report["source_explicit"].values())
report["artifact_policy_preserved"] = any(report["artifact_explicit_counts"].values())
print(json.dumps(report, indent=2, sort_keys=True))
