#!/usr/bin/env python3
"""Audit Linux real-time scheduler throttling policy and Laguna provenance."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_optional(path: pathlib.Path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


sysctl = pathlib.Path("/proc/sys/kernel")
period = read_optional(sysctl / "sched_rt_period_us")
runtime = read_optional(sysctl / "sched_rt_runtime_us")
env_text = (ROOT / "env.sh").read_text().lower()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text().lower()
terms = ("sched_rt_period_us", "sched_rt_runtime_us", "rt throttling", "rt-throttling")
report = {
    "angle": "linux-real-time-scheduler-throttling-policy-and-provenance",
    "live": {
        "sched_rt_period_us": int(period) if period is not None else None,
        "sched_rt_runtime_us": int(runtime) if runtime is not None else None,
        "global_rt_bandwidth_fraction": (
            int(runtime) / int(period)
            if period is not None and runtime is not None and int(runtime) >= 0
            else None
        ),
        "rt_throttling_disabled": runtime == "-1",
    },
    "provenance": {
        "env_controls_rt_throttling": any(term in env_text for term in terms),
        "serial_harness_records_rt_throttling": any(term in harness_text for term in terms),
    },
}
out = ROOT / "benchmark/results/kernel-rt-scheduler-throttling-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
