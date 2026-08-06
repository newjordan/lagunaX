#!/usr/bin/env python3
"""Audit scheduler autogroup, CFS bandwidth, and RR timeslice policy provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/kernel-scheduler-timeslice-autogroup-audit-20260807.json"


def read_int(path: str) -> int:
    try:
        return int(Path(path).read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise RuntimeError(f"cannot read integer kernel control {path}: {exc}") from exc


controls = {
    "sched_autogroup_enabled": read_int("/proc/sys/kernel/sched_autogroup_enabled"),
    "sched_cfs_bandwidth_slice_us": read_int("/proc/sys/kernel/sched_cfs_bandwidth_slice_us"),
    "sched_rr_timeslice_ms": read_int("/proc/sys/kernel/sched_rr_timeslice_ms"),
}
env_text = ENV.read_text(errors="replace")
harness_text = HARNESS.read_text(errors="replace")
source = env_text + "\n" + harness_text
policy_pattern = re.compile(
    r"sched_autogroup_enabled|sched_cfs_bandwidth_slice_us|sched_rr_timeslice_ms", re.I
)
metrics_pattern = re.compile(
    r"/proc/sys/kernel/sched_(?:autogroup_enabled|cfs_bandwidth_slice_us|rr_timeslice_ms)", re.I
)

report = {
    "angle": "kernel-scheduler-timeslice-autogroup-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": controls,
    "laguna": {
        "env_or_harness_controls_audited_scheduler_policy": bool(policy_pattern.search(source)),
        "serial_harness_records_audited_scheduler_policy": bool(metrics_pattern.search(harness_text)),
    },
    "interpretation": (
        "Autogrouping changes fair-scheduler task grouping; the CFS bandwidth slice controls "
        "quota transfer granularity; and the RR timeslice controls SCHED_RR quantum length."
    ),
}
assert controls["sched_autogroup_enabled"] in (0, 1)
assert controls["sched_cfs_bandwidth_slice_us"] > 0
assert controls["sched_rr_timeslice_ms"] > 0
assert not report["laguna"]["env_or_harness_controls_audited_scheduler_policy"]
assert not report["laguna"]["serial_harness_records_audited_scheduler_policy"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
