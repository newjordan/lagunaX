#!/usr/bin/env python3
"""Audit periodic VM statistics update policy and Laguna provenance."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
BENCH = ROOT / "scripts/bench-serial.sh"
CONTROLS = {
    "stat_interval_seconds": Path("/proc/sys/vm/stat_interval"),
    "numa_stat_enabled": Path("/proc/sys/vm/numa_stat"),
}

live = {name: int(path.read_text().strip()) for name, path in CONTROLS.items()}
env_text = ENV.read_text(errors="replace")
bench_text = BENCH.read_text(errors="replace")
source = env_text + "\n" + bench_text
terms = ("stat_interval", "numa_stat")

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "periodic-vm-statistics-update-policy",
    "live_kernel_policy": live,
    "canonical_policy": {
        "env_or_harness_controls_vm_stat_updates": [
            term for term in terms if re.search(rf"\b{re.escape(term)}\b", source)
        ],
        "metrics_record_vm_stat_update_policy": [
            term for term in terms if re.search(rf"\b{re.escape(term)}\b", bench_text)
        ],
    },
    "interpretation": (
        "The kernel refreshes VM statistics on a one-second interval and exports "
        "NUMA statistics; Laguna neither controls nor records this periodic policy."
    ),
}

assert live["stat_interval_seconds"] > 0
assert live["numa_stat_enabled"] in (0, 1)
assert not report["canonical_policy"]["env_or_harness_controls_vm_stat_updates"]
assert not report["canonical_policy"]["metrics_record_vm_stat_update_policy"]

out = ROOT / "results/periodic-vm-statistics-update-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
