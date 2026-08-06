#!/usr/bin/env python3
"""Audit explicit multi-device model placement and tensor-splitting coverage."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
BENCH = ROOT / "scripts" / "bench-serial.sh"
RESULTS = ROOT / "results"
KNOBS = ("--split-mode", "--tensor-split", "--main-gpu", "--device")

sources = {str(p.relative_to(ROOT)): p.read_text(errors="replace") for p in (ENV, BENCH)}
artifacts = []
for path in RESULTS.rglob("*"):
    if path.name.startswith("multi-device-placement-audit-"):
        continue
    if path.is_file() and path.suffix.lower() in {".json", ".txt", ".log", ".md"}:
        try:
            artifacts.append(path.read_text(errors="replace"))
        except OSError:
            pass

# Resolve the benchmark exactly as env.sh does and retain command failure explicitly.
proc = subprocess.run(
    ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; "$LX_BIN/llama-bench" --list-devices'],
    cwd=ROOT, text=True, capture_output=True, check=False,
)
devices = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
report = {
    "policy": "multi_device_tensor_placement",
    "knobs": list(KNOBS),
    "source_explicit": {k: [name for name, text in sources.items() if k in text] for k in KNOBS},
    "artifact_explicit_counts": {k: sum(k in text for text in artifacts) for k in KNOBS},
    "artifacts_scanned": len(artifacts),
    "list_devices_exit_code": proc.returncode,
    "device_inventory": devices,
    "device_inventory_count": len(devices),
}
report["serial_policy_explicit"] = any(report["source_explicit"].values())
print(json.dumps(report, indent=2, sort_keys=True))
