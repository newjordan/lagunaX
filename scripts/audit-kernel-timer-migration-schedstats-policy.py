#!/usr/bin/env python3
"""Audit kernel timer migration and scheduler-statistics policy provenance."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/kernel-timer-migration-schedstats-audit-20260807.json"


def read_int(path: str) -> int:
    return int(Path(path).read_text().strip())


timer_migration = read_int("/proc/sys/kernel/timer_migration")
schedstats = read_int("/proc/sys/kernel/sched_schedstats")
source = "\n".join(
    p.read_text(errors="replace")
    for p in (ROOT / "env.sh", ROOT / "scripts/bench-serial.sh")
)
metrics = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
controls = ("timer_migration", "sched_schedstats")
artifact = {
    "live": {
        "kernel_timer_migration": timer_migration,
        "kernel_sched_schedstats": schedstats,
    },
    "laguna": {
        "env_or_harness_controls_policy": any(
            re.search(rf"\b{re.escape(name)}\b", source) for name in controls
        ),
        "serial_metrics_record_policy": any(name in metrics for name in controls),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(OUT)
