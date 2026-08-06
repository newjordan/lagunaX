#!/usr/bin/env python3
"""Audit scheduler utilization-clamping policy and Laguna provenance."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSCTL = pathlib.Path("/proc/sys/kernel")
KNOBS = (
    "sched_util_clamp_min",
    "sched_util_clamp_max",
    "sched_util_clamp_min_rt_default",
)


def read_optional(path: pathlib.Path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
values = {name: read_optional(SYSCTL / name) for name in KNOBS}
report = {
    "angle": "kernel-scheduler-utilization-clamping-policy",
    "active_policy": values,
    "available_knobs": [name for name, value in values.items() if value is not None],
    "provenance": {
        "env_controls_any_uclamp_knob": any(name in env_text for name in KNOBS),
        "serial_harness_records_any_uclamp_knob": any(
            name in harness_text for name in KNOBS
        ),
    },
}
out = ROOT / "benchmark/results/kernel-scheduler-utilization-clamping-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
