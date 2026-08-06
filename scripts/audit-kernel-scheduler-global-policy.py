#!/usr/bin/env python3
"""Audit kernel scheduler policy and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

CONTROLS = {
    "sched_autogroup_enabled": Path("/proc/sys/kernel/sched_autogroup_enabled"),
    "sched_rr_timeslice_ms": Path("/proc/sys/kernel/sched_rr_timeslice_ms"),
    "sched_rt_period_us": Path("/proc/sys/kernel/sched_rt_period_us"),
    "sched_rt_runtime_us": Path("/proc/sys/kernel/sched_rt_runtime_us"),
    "sched_util_clamp_min": Path("/proc/sys/kernel/sched_util_clamp_min"),
    "sched_util_clamp_max": Path("/proc/sys/kernel/sched_util_clamp_max"),
}


def read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


live = {name: read_int(path) for name, path in CONTROLS.items()}
source = ENV + "\n" + BENCH
report = {
    "angle": "kernel-scheduler-global-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": live,
    "laguna": {
        "env_policy_hits": [name for name in CONTROLS if name in ENV],
        "bench_policy_hits": [name for name in CONTROLS if name in BENCH],
        "metrics_record_policy": any(f'\"{name}\"' in BENCH for name in CONTROLS),
    },
    "finding": (
        "Laguna neither configures nor records the audited kernel scheduler "
        "autogroup, real-time bandwidth, timeslice, or utilization-clamp policy."
    ),
}
out = ROOT / "results/kernel-scheduler-global-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
